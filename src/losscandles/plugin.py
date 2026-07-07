"""LossCandlesPlugin: a TensorBoard dynamic plugin rendering loss as candlesticks.

Reads scalar data through the modern `data_provider` API (never the deprecated
EventMultiplexer directly), so it works on any logdir written by plain
`SummaryWriter.add_scalar` with zero instrumentation. Optional enrichment
(epoch gridlines, a declared volume tag) is read from summaries tagged with our
own plugin_name, written by `losscandles.torch.LossCandlesWriter`.
"""

from __future__ import annotations

import json
import math
import os

import werkzeug
import werkzeug.exceptions
from werkzeug import wrappers

from tensorboard import context as tb_context
from tensorboard import plugin_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin

from losscandles import metadata
from losscandles.annotations import detect_circuit_breakers, detect_flash_crashes, detect_rate_cuts
from losscandles.indicators import overfit_regions as compute_overfit_regions
from losscandles.ohlc import aggregate_ohlc, align_candles

SCALARS_PLUGIN_NAME = "scalars"
# read_scalars(downsample=...) only limits how much of what TensorBoard's own
# ingestion already retained gets returned to us -- it cannot recover points
# TensorBoard dropped at ingestion time. That earlier, much lower cap (~1000
# points per tag by default) is set by the `tensorboard` CLI's own
# --samples_per_plugin flag, outside this plugin's control; see the README's
# "Order book depth" section. We still pass a generous value here so we never
# add our own additional truncation on top of whatever survived ingestion.
_DOWNSAMPLE_CAP = 1_000_000


class LossCandlesPlugin(base_plugin.TBPlugin):
    plugin_name = metadata.PLUGIN_NAME

    def __init__(self, context):
        self._data_provider = context.data_provider

    def get_plugin_apps(self):
        return {
            "/index.js": self._serve_js,
            "/tags": self._serve_tags,
            "/ohlc": self._serve_ohlc,
            "/annotations": self._serve_annotations,
        }

    def is_active(self):
        if not self._data_provider:
            return False
        try:
            mapping = self._data_provider.list_scalars(
                tb_context.RequestContext(), experiment_id="", plugin_name=SCALARS_PLUGIN_NAME
            )
        except Exception:
            return False
        return any(mapping.values())

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(es_module_path="/index.js", tab_name="Candles")

    @wrappers.Request.application
    def _serve_js(self, request):
        del request
        filepath = os.path.join(os.path.dirname(__file__), "static", "index.js")
        with open(filepath) as infile:
            contents = infile.read()
        return werkzeug.Response(contents, content_type="text/javascript")

    def _list_scalar_tags(self, ctx, experiment):
        mapping = self._data_provider.list_scalars(ctx, experiment_id=experiment, plugin_name=SCALARS_PLUGIN_NAME)
        return {run: sorted(tags.keys()) for run, tags in mapping.items()}

    def _read_config(self, ctx, experiment):
        mapping = self._data_provider.list_scalars(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME)
        config = {}
        for run, tags in mapping.items():
            md = tags.get(metadata.TAG_CONFIG)
            if md is not None:
                config[run] = metadata.parse_config_content(md.plugin_content)
        return config

    @wrappers.Request.application
    def _serve_tags(self, request):
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        body = {
            "runs": self._list_scalar_tags(ctx, experiment),
            "config": self._read_config(ctx, experiment),
        }
        return werkzeug.Response(json.dumps(body), content_type="application/json")

    def _read_series(self, ctx, experiment, run, tag, plugin_name=SCALARS_PLUGIN_NAME):
        if not run or not tag:
            return None
        data = self._data_provider.read_scalars(
            ctx,
            experiment_id=experiment,
            plugin_name=plugin_name,
            downsample=_DOWNSAMPLE_CAP,
            run_tag_filter=provider.RunTagFilter(runs=[run], tags=[tag]),
        )
        series = data.get(run, {}).get(tag)
        if not series:
            return None
        return [(x.step, x.value) for x in series]

    @wrappers.Request.application
    def _serve_ohlc(self, request):
        run = request.args.get("run")
        tag = request.args.get("tag")
        if not run or not tag:
            raise werkzeug.exceptions.BadRequest("Must specify run and tag")
        window = _positive_int(request.args.get("window", "100"), "window")
        volume_tag = request.args.get("volume_tag") or None

        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)

        series = self._read_series(ctx, experiment, run, tag)
        if series is None:
            raise werkzeug.exceptions.NotFound(f"No scalar data for run={run!r}, tag={tag!r}")
        steps = [s for s, _ in series]
        values = [v for _, v in series]

        volume_values = None
        if volume_tag:
            vol_series = self._read_series(ctx, experiment, run, volume_tag)
            if vol_series is not None:
                vol_map = dict(vol_series)
                volume_values = [vol_map.get(s, float("nan")) for s in steps]

        candles = aggregate_ohlc(steps, values, window=window, volume_values=volume_values)
        body = {"candles": [_candle_json(c) for c in candles]}
        return werkzeug.Response(json.dumps(body), content_type="application/json")

    @wrappers.Request.application
    def _serve_annotations(self, request):
        run = request.args.get("run")
        tag = request.args.get("tag")
        if not run or not tag:
            raise werkzeug.exceptions.BadRequest("Must specify run and tag")
        window = _positive_int(request.args.get("window", "100"), "window")
        lr_tag = request.args.get("lr_tag") or None
        val_tag = request.args.get("val_tag") or None
        flash_crash_threshold = _positive_float(request.args.get("flash_crash_threshold", "0.30"), "flash_crash_threshold")
        rate_cut_threshold = _positive_float(request.args.get("rate_cut_threshold", "0.10"), "rate_cut_threshold")
        min_consecutive = _positive_int(request.args.get("min_consecutive", "3"), "min_consecutive")

        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)

        series = self._read_series(ctx, experiment, run, tag)
        if series is None:
            raise werkzeug.exceptions.NotFound(f"No scalar data for run={run!r}, tag={tag!r}")
        steps = [s for s, _ in series]
        values = [v for _, v in series]
        candles = aggregate_ohlc(steps, values, window=window)

        circuit_breakers = [_annotation_json(a, candles) for a in detect_circuit_breakers(candles)]
        flash_crashes = [
            _annotation_json(a, candles)
            for a in detect_flash_crashes(candles, threshold=flash_crash_threshold)
        ]

        rate_cuts = []
        if lr_tag:
            lr_series = self._read_series(ctx, experiment, run, lr_tag)
            if lr_series is not None:
                lr_candles = aggregate_ohlc([s for s, _ in lr_series], [v for _, v in lr_series], window=window)
                rate_cuts = [
                    _annotation_json(a, lr_candles)
                    for a in detect_rate_cuts(lr_candles, min_drop_frac=rate_cut_threshold)
                ]

        regions = []
        if val_tag:
            val_series = self._read_series(ctx, experiment, run, val_tag)
            if val_series is not None:
                val_candles = aggregate_ohlc([s for s, _ in val_series], [v for _, v in val_series], window=window)
                # Align by step range, not list position: train/val are often
                # logged at different frequencies (e.g. val once per epoch).
                pairs = align_candles(candles, val_candles, window)
                if pairs:
                    train_closes = [p[0].close for p in pairs]
                    val_closes = [p[1].close for p in pairs]
                    for start, end in compute_overfit_regions(train_closes, val_closes, min_consecutive=min_consecutive):
                        regions.append(
                            {
                                "start_index": pairs[start][0].index,
                                "end_index": pairs[end][0].index,
                                "start_step": pairs[start][0].start_step,
                                "end_step": pairs[end][0].end_step,
                                "label": "OVERFITTING",
                            }
                        )

        body = {
            "circuit_breakers": circuit_breakers,
            "flash_crashes": flash_crashes,
            "rate_cuts": rate_cuts,
            "overfit_regions": regions,
            "epoch_boundaries": self._epoch_boundaries(ctx, experiment, run),
        }
        return werkzeug.Response(json.dumps(body), content_type="application/json")

    def _epoch_boundaries(self, ctx, experiment, run):
        # Zero-instrumentation detection: scans the "epoch" scalar tag for value
        # transitions. Reported steps are only exact if TensorBoard retained every
        # point for that tag (see the _DOWNSAMPLE_CAP note above) -- on a run that
        # exceeds TB's own ingestion cap, a transition is reported at the next
        # *surviving* sample after the true boundary, not the boundary itself.
        # add_epoch_boundary() markers (below) are exact regardless, since those
        # are few enough to never hit that cap in practice.
        boundaries = set()
        epoch_series = self._read_series(ctx, experiment, run, "epoch")
        if epoch_series:
            prev = None
            for step, epoch in epoch_series:
                if prev is not None and epoch != prev:
                    boundaries.add(int(step))
                prev = epoch
        marker_series = self._read_series(ctx, experiment, run, metadata.TAG_EPOCH_BOUNDARY, plugin_name=metadata.PLUGIN_NAME)
        if marker_series:
            boundaries.update(int(step) for step, _ in marker_series)
        return sorted(boundaries)


def _positive_int(raw: str, name: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise werkzeug.exceptions.BadRequest(f"{name} must be an integer, got {raw!r}")
    if value < 1:
        raise werkzeug.exceptions.BadRequest(f"{name} must be >= 1, got {value}")
    return value


def _positive_float(raw, name: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise werkzeug.exceptions.BadRequest(f"{name} must be a number, got {raw!r}")
    if not math.isfinite(value) or value <= 0:
        raise werkzeug.exceptions.BadRequest(f"{name} must be a finite number > 0, got {raw!r}")
    return value


def _json_safe(x: float):
    return None if x != x or x in (float("inf"), float("-inf")) else x


def _candle_json(c) -> dict:
    return {
        "index": c.index,
        "step_start": c.start_step,
        "step_end": c.end_step,
        "open": _json_safe(c.open),
        "high": _json_safe(c.high),
        "low": _json_safe(c.low),
        "close": _json_safe(c.close),
        "volume": _json_safe(c.volume),
        "n_points": c.n_points,
        "has_nonfinite": c.has_nonfinite,
    }


def _annotation_json(annotation, candles) -> dict:
    return {"index": annotation.index, "step": candles[annotation.index].start_step, "label": annotation.label}
