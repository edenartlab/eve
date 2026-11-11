.PHONY: help price price-prod price-check price-sync test

DB ?= PROD

help:
	@echo "Tool pricing"
	@echo "  make price TOOL=flux_dev   what does one tool cost, and does the repo agree?"
	@echo "  make price                 audit every tool against PROD"
	@echo "  make price-check           fail if the repo and PROD disagree"
	@echo "  make price-sync            push the repo's prices to PROD (asks first)"
	@echo ""
	@echo "  To change a price: edit cost_estimate in the tool's api.yaml and merge."
	@echo ""
	@echo "Tests"
	@echo "  make test                  run the offline test suite"

## What does this tool cost right now, and does the repo agree?
## Usage: make price TOOL=flux_dev [DB=PROD]
price:
	eve tool price --db $(DB) $(TOOL)

price-check:
	eve tool price-check --db PROD

price-sync:
	eve tool price-sync --db PROD

test:
	pytest tests/ -q -m "not live and not integration"
