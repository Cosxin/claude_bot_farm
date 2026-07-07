import {
  createChart,
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  createSeriesMarkers,
  IChartApi,
  ISeriesApi,
  SeriesMarker,
  Time,
  UTCTimestamp,
} from "lightweight-charts";

interface Candle {
  index: number;
  step_start: number;
  step_end: number;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  volume: number | null;
  n_points: number;
  has_nonfinite: boolean;
}

interface OhlcResponse {
  candles: Candle[];
}

interface Annotation {
  index: number;
  step: number;
  label: string;
}

interface OverfitRegion {
  start_index: number;
  end_index: number;
  start_step: number;
  end_step: number;
  label: string;
}

interface AnnotationsResponse {
  circuit_breakers: Annotation[];
  flash_crashes: Annotation[];
  rate_cuts: Annotation[];
  overfit_regions: OverfitRegion[];
  epoch_boundaries: number[];
}

interface TagsResponse {
  runs: Record<string, string[]>;
  config: Record<string, { volume_tag?: string }>;
}

const BULLISH_GREEN = "#26a69a";
const BEARISH_RED = "#ef5350";
const CIRCUIT_BREAKER_COLOR = "#111111";
const RATE_CUT_COLOR = "#7c4dff";
const OVERFIT_COLOR = "#c99a06";
const SMA_PALETTE = ["#1f77b4", "#2ca02c", "#9467bd", "#17becf"];
const EMA_PALETTE = ["#ff7f0e", "#d62728", "#8c564b", "#e377c2"];

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  children: (Node | string)[] = []
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else node.setAttribute(k, v);
  }
  for (const child of children) {
    node.append(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

function guessValTag(trainTag: string, tags: string[]): string | null {
  const candidates = [trainTag.replace(/train/i, "val"), trainTag.replace(/training/i, "validation")];
  for (const c of candidates) {
    if (c !== trainTag && tags.includes(c)) return c;
  }
  return null;
}

function guessLrTag(tags: string[]): string | null {
  if (tags.includes("lr")) return "lr";
  return tags.find((t) => /(^|\/)lr$/i.test(t) || /learning_rate/i.test(t)) ?? null;
}

function guessVolumeTag(tags: string[]): string | null {
  // Point-count volume is flat/uninformative when logging cadence is constant
  // (the common case), so prefer a real activity signal if one is logged --
  // grad norm is the best analog to "how hard the market moved."
  const patterns = [/grad.?norm/i, /gradient.?norm/i, /tokens?[_.]?(per[_.]?sec|\/sec)/i, /samples?[_.]?(per[_.]?sec|\/sec)/i, /throughput/i];
  for (const p of patterns) {
    const hit = tags.find((t) => p.test(t));
    if (hit) return hit;
  }
  return null;
}

async function fetchJson<T>(url: string): Promise<T | null> {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

function sma(values: (number | null)[], window: number): (number | null)[] {
  const out: (number | null)[] = new Array(values.length).fill(null);
  for (let i = window - 1; i < values.length; i++) {
    const chunk = values.slice(i - window + 1, i + 1).filter((v): v is number => v !== null);
    if (chunk.length > 0) out[i] = chunk.reduce((a, b) => a + b, 0) / chunk.length;
  }
  return out;
}

function ema(values: (number | null)[], span: number): (number | null)[] {
  const alpha = 2 / (span + 1);
  const out: (number | null)[] = [];
  let prev: number | null = null;
  for (const v of values) {
    if (v === null) {
      out.push(prev);
      continue;
    }
    prev = prev === null ? v : alpha * v + (1 - alpha) * prev;
    out.push(prev);
  }
  return out;
}

function bollinger(values: (number | null)[], window: number, numStd: number) {
  const mid: (number | null)[] = new Array(values.length).fill(null);
  const upper: (number | null)[] = new Array(values.length).fill(null);
  const lower: (number | null)[] = new Array(values.length).fill(null);
  for (let i = window - 1; i < values.length; i++) {
    const chunk = values.slice(i - window + 1, i + 1).filter((v): v is number => v !== null);
    if (chunk.length > 0) {
      const mean = chunk.reduce((a, b) => a + b, 0) / chunk.length;
      const variance = chunk.reduce((a, b) => a + (b - mean) ** 2, 0) / chunk.length;
      const std = Math.sqrt(variance);
      mid[i] = mean;
      upper[i] = mean + numStd * std;
      lower[i] = mean - numStd * std;
    }
  }
  return { mid, upper, lower };
}

function toLineData(times: UTCTimestamp[], values: (number | null)[]): { time: UTCTimestamp; value: number }[] {
  const out: { time: UTCTimestamp; value: number }[] = [];
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v !== null && Number.isFinite(v)) out.push({ time: times[i], value: v });
  }
  return out;
}

function parseWindowList(raw: string): number[] {
  return raw
    .split(",")
    .map((s) => parseInt(s.trim(), 10))
    .filter((n) => Number.isFinite(n) && n > 1);
}

class CandlesApp {
  private root: HTMLElement;
  private chartContainer: HTMLElement;
  private statusEl: HTMLElement;
  private chart: IChartApi | null = null;
  private resizeHandler: (() => void) | null = null;

  private runSelect = el("select", {});
  private tagSelect = el("select", {});
  private windowInput = el("input", { type: "number", min: "1", value: "45" }) as HTMLInputElement;
  private smaInput = el("input", { type: "text", placeholder: "e.g. 5,20" }) as HTMLInputElement;
  private emaInput = el("input", { type: "text", placeholder: "e.g. 10" }) as HTMLInputElement;
  private bollingerCheck = el("input", { type: "checkbox" }) as HTMLInputElement;
  private overfitCheck = el("input", { type: "checkbox" }) as HTMLInputElement;
  private wallStreetCheck = el("input", { type: "checkbox" }) as HTMLInputElement;
  private liveCheck = el("input", { type: "checkbox" }) as HTMLInputElement;
  private refreshButton = el("button", {}, ["Refresh"]);

  private tagsData: TagsResponse = { runs: {}, config: {} };
  private pollHandle: number | null = null;

  constructor(root: HTMLElement) {
    this.root = root;
    this.statusEl = el("div", { class: "lc-status" }, ["Loading…"]);
    this.chartContainer = el("div", { class: "lc-chart" });
    this.buildLayout();
  }

  private labeled(text: string, control: HTMLElement): HTMLElement {
    return el("label", {}, [text, control]);
  }

  private checkboxLabel(text: string, control: HTMLElement): HTMLElement {
    return el("label", { class: "lc-checkbox" }, [control, text]);
  }

  private buildLayout(): void {
    const controls = el("div", { class: "lc-controls" }, [
      this.labeled("Run", this.runSelect),
      this.labeled("Tag", this.tagSelect),
      this.labeled("Window", this.windowInput),
      this.labeled("SMA", this.smaInput),
      this.labeled("EMA", this.emaInput),
      this.checkboxLabel("Bollinger", this.bollingerCheck),
      this.checkboxLabel("Overfit index", this.overfitCheck),
      this.checkboxLabel("Wall Street mode", this.wallStreetCheck),
      this.checkboxLabel("Live (30s)", this.liveCheck),
      this.refreshButton,
    ]);

    const style = el("style", {}, [
      `
      .lc-app { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 12px; color: #1a1a1a; }
      .lc-app h2 { margin: 0 0 10px 0; font-size: 16px; }
      .lc-controls { display: flex; flex-wrap: wrap; gap: 14px; align-items: end; margin-bottom: 10px; font-size: 13px; }
      .lc-controls label { display: flex; flex-direction: column; gap: 2px; font-size: 11px; color: #555; }
      .lc-controls input[type=number], .lc-controls input[type=text] { width: 5.5em; }
      .lc-controls .lc-checkbox { flex-direction: row; align-items: center; gap: 4px; }
      .lc-status { font-size: 12px; color: #888; margin-bottom: 6px; min-height: 1.2em; }
      .lc-chart { border: 1px solid #e0e0e0; border-radius: 4px; }
      button { cursor: pointer; }
      `,
    ]);

    this.root.append(
      style,
      el("div", { class: "lc-app" }, [
        el("h2", {}, ["\u{1F56F}\u{FE0F} losscandles — loss curves and loss curves are different graphs. For now."]),
        controls,
        this.statusEl,
        this.chartContainer,
      ])
    );

    this.runSelect.addEventListener("change", () => this.onRunChange());
    for (const control of [this.tagSelect, this.windowInput, this.smaInput, this.emaInput, this.bollingerCheck, this.overfitCheck, this.wallStreetCheck]) {
      control.addEventListener("change", () => this.refresh());
    }
    this.refreshButton.addEventListener("click", () => this.refresh());
    this.liveCheck.addEventListener("change", () => this.togglePolling());
  }

  async start(): Promise<void> {
    await this.loadTags();
    await this.refresh();
  }

  private async loadTags(): Promise<void> {
    const data = await fetchJson<TagsResponse>("tags");
    this.tagsData = data ?? { runs: {}, config: {} };
    const runs = Object.keys(this.tagsData.runs).sort();
    this.runSelect.replaceChildren(...runs.map((r) => el("option", { value: r }, [r])));
    if (runs.length > 0) this.runSelect.value = runs[0];
    this.populateTagsForRun();
  }

  private populateTagsForRun(): void {
    const run = this.runSelect.value;
    const tags = this.tagsData.runs[run] ?? [];
    this.tagSelect.replaceChildren(...tags.map((t) => el("option", { value: t }, [t])));
    const preferred = tags.find((t) => /loss/i.test(t)) ?? tags[0];
    if (preferred) this.tagSelect.value = preferred;
  }

  private onRunChange(): void {
    this.populateTagsForRun();
    this.refresh();
  }

  private togglePolling(): void {
    if (this.pollHandle !== null) {
      window.clearInterval(this.pollHandle);
      this.pollHandle = null;
    }
    if (this.liveCheck.checked) {
      this.pollHandle = window.setInterval(() => this.refresh(), 30_000);
    }
  }

  private async refresh(): Promise<void> {
    const run = this.runSelect.value;
    const tag = this.tagSelect.value;
    if (!run || !tag) {
      this.statusEl.textContent = "No scalar data found in this logdir yet.";
      return;
    }
    const windowSize = Math.max(1, parseInt(this.windowInput.value, 10) || 100);
    const tags = this.tagsData.runs[run] ?? [];
    const valTag = this.overfitCheck.checked ? guessValTag(tag, tags) : null;
    const lrTag = guessLrTag(tags);
    const volumeTag = this.tagsData.config[run]?.volume_tag ?? guessVolumeTag(tags);

    this.statusEl.textContent = "Loading…";

    const ohlcParams = new URLSearchParams({ run, tag, window: String(windowSize) });
    if (volumeTag) ohlcParams.set("volume_tag", volumeTag);
    const annParams = new URLSearchParams({ run, tag, window: String(windowSize) });
    if (lrTag) annParams.set("lr_tag", lrTag);
    if (valTag) annParams.set("val_tag", valTag);

    const [ohlc, annotations, valOhlc] = await Promise.all([
      fetchJson<OhlcResponse>(`ohlc?${ohlcParams}`),
      fetchJson<AnnotationsResponse>(`annotations?${annParams}`),
      valTag
        ? fetchJson<OhlcResponse>(`ohlc?${new URLSearchParams({ run, tag: valTag, window: String(windowSize) })}`)
        : Promise.resolve(null),
    ]);

    if (!ohlc || ohlc.candles.length === 0) {
      this.statusEl.textContent = `No data yet for run=${run} tag=${tag}.`;
      return;
    }
    const last = ohlc.candles[ohlc.candles.length - 1];
    this.statusEl.textContent =
      `${ohlc.candles.length} candles · steps ${ohlc.candles[0].step_start}–${last.step_end}` +
      (valTag ? ` · val: ${valTag}` : "") +
      (lrTag ? ` · lr: ${lrTag}` : "") +
      (volumeTag ? ` · volume: ${volumeTag}` : " · volume: point count");

    this.render(ohlc.candles, annotations, valTag ? valOhlc?.candles ?? null : null, windowSize);
  }

  private render(candles: Candle[], annotations: AnnotationsResponse | null, valCandles: Candle[] | null, windowSize: number): void {
    if (this.resizeHandler) window.removeEventListener("resize", this.resizeHandler);
    if (this.chart) {
      this.chart.remove();
      this.chart = null;
    }
    this.chartContainer.replaceChildren();

    const showOverfit = valCandles !== null && valCandles.length > 0;
    const totalHeight = showOverfit ? 640 : 500;
    this.chartContainer.style.height = `${totalHeight}px`;

    const chart = createChart(this.chartContainer, {
      width: this.chartContainer.clientWidth || 900,
      height: totalHeight,
      layout: { textColor: "#333", background: { color: "white" } },
      timeScale: { tickMarkFormatter: (time: Time) => String(time), borderVisible: true },
      localization: { timeFormatter: (time: Time) => `step ${time}` },
    });
    this.chart = chart;

    const wallStreet = this.wallStreetCheck.checked;
    const upColor = wallStreet ? BULLISH_GREEN : BEARISH_RED;
    const downColor = wallStreet ? BEARISH_RED : BULLISH_GREEN;

    const mainPane = 0;
    const candleSeries = chart.addSeries(
      CandlestickSeries,
      {
        upColor,
        downColor,
        borderUpColor: upColor,
        borderDownColor: downColor,
        wickUpColor: upColor,
        wickDownColor: downColor,
      },
      mainPane
    );
    candleSeries.setData(
      candles
        .filter((c): c is Candle & { open: number; high: number; low: number; close: number } => c.open !== null)
        .map((c) => ({
          time: c.step_start as unknown as UTCTimestamp,
          open: c.open,
          high: c.high,
          low: c.low,
          close: c.close,
        }))
    );

    this.addIndicators(chart, candles, mainPane);

    const volumePane = 1;
    const volumeSeries = chart.addSeries(HistogramSeries, { color: "#88888844", priceFormat: { type: "volume" } }, volumePane);
    volumeSeries.setData(
      candles
        .filter((c) => c.volume !== null && c.open !== null && c.close !== null)
        .map((c) => ({
          time: c.step_start as unknown as UTCTimestamp,
          value: c.volume as number,
          color: (c.close as number) >= (c.open as number) ? upColor : downColor,
        }))
    );
    chart.panes()[volumePane]?.setHeight(showOverfit ? 110 : 140);

    if (showOverfit && valCandles) {
      this.addOverfitPane(chart, candles, valCandles, annotations, windowSize);
    }

    if (annotations) this.addMarkers(candleSeries, annotations);

    chart.timeScale().fitContent();
    this.resizeHandler = () => chart.applyOptions({ width: this.chartContainer.clientWidth || 900 });
    window.addEventListener("resize", this.resizeHandler);
  }

  private addIndicators(chart: IChartApi, candles: Candle[], pane: number): void {
    const closes = candles.map((c) => c.close);
    const times = candles.map((c) => c.step_start as unknown as UTCTimestamp);

    parseWindowList(this.smaInput.value).forEach((w, i) => {
      const series = chart.addSeries(LineSeries, { color: SMA_PALETTE[i % SMA_PALETTE.length], lineWidth: 1, title: `SMA(${w})` }, pane);
      series.setData(toLineData(times, sma(closes, w)));
    });
    parseWindowList(this.emaInput.value).forEach((w, i) => {
      const series = chart.addSeries(
        LineSeries,
        { color: EMA_PALETTE[i % EMA_PALETTE.length], lineWidth: 1, lineStyle: 2, title: `EMA(${w})` },
        pane
      );
      series.setData(toLineData(times, ema(closes, w)));
    });
    if (this.bollingerCheck.checked) {
      const { mid, upper, lower } = bollinger(closes, 20, 2);
      const dim = { lineWidth: 1 as const, color: "rgba(120,120,120,0.55)" };
      chart.addSeries(LineSeries, { ...dim, title: "Bollinger mid" }, pane).setData(toLineData(times, mid));
      chart.addSeries(LineSeries, { ...dim, title: "Bollinger upper" }, pane).setData(toLineData(times, upper));
      chart.addSeries(LineSeries, { ...dim, title: "Bollinger lower" }, pane).setData(toLineData(times, lower));
    }
  }

  private addOverfitPane(
    chart: IChartApi,
    candles: Candle[],
    valCandles: Candle[],
    annotations: AnnotationsResponse | null,
    windowSize: number
  ): void {
    const pane = 2;
    // Align by step range, not list position: train/val are often logged at
    // different frequencies (e.g. val once per epoch, train every step), so
    // candle i in one series need not cover the same steps as candle i in
    // the other.
    const valByWindowId = new Map(valCandles.map((c) => [Math.floor(c.step_start / windowSize), c]));
    const spreadData: { time: UTCTimestamp; value: number }[] = [];
    for (const c of candles) {
      const v = valByWindowId.get(Math.floor(c.step_start / windowSize));
      if (c.close !== null && v?.close != null) {
        spreadData.push({ time: c.step_start as unknown as UTCTimestamp, value: v.close - c.close });
      }
    }
    const spreadSeries = chart.addSeries(LineSeries, { color: OVERFIT_COLOR, lineWidth: 2, title: "val − train" }, pane);
    spreadSeries.setData(spreadData);
    chart.panes()[pane]?.setHeight(110);

    if (annotations?.overfit_regions.length) {
      const markers: SeriesMarker<Time>[] = annotations.overfit_regions.map((r) => ({
        time: r.start_step as unknown as UTCTimestamp,
        position: "aboveBar" as const,
        color: OVERFIT_COLOR,
        shape: "square" as const,
        text: "OVERFITTING",
      }));
      createSeriesMarkers(spreadSeries, markers);
    }
  }

  private addMarkers(series: ISeriesApi<"Candlestick">, annotations: AnnotationsResponse): void {
    const markers: SeriesMarker<Time>[] = [];
    for (const a of annotations.circuit_breakers) {
      markers.push({ time: a.step as unknown as UTCTimestamp, position: "aboveBar", color: CIRCUIT_BREAKER_COLOR, shape: "arrowDown", text: "\u{1F6D1} HALT" });
    }
    for (const a of annotations.flash_crashes) {
      markers.push({ time: a.step as unknown as UTCTimestamp, position: "aboveBar", color: BEARISH_RED, shape: "circle", size: 0, text: "S" });
    }
    for (const a of annotations.rate_cuts) {
      markers.push({ time: a.step as unknown as UTCTimestamp, position: "belowBar", color: RATE_CUT_COLOR, shape: "circle", size: 0, text: "CUT" });
    }
    for (const step of annotations.epoch_boundaries) {
      markers.push({ time: step as unknown as UTCTimestamp, position: "inBar", color: "#999999", shape: "square", text: "" });
    }
    markers.sort((a, b) => (a.time as unknown as number) - (b.time as unknown as number));
    createSeriesMarkers(series, markers);
  }
}

export async function render(): Promise<void> {
  const app = new CandlesApp(document.body);
  await app.start();
}
