.PHONY: frontend test

frontend:
	cd frontend && npm install && npm run build

test:
	pytest
