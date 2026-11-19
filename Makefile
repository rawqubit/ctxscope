.PHONY: test demo report examples clean

test:
	python3 -m unittest discover -s tests -v

demo:
	python3 ctxscope.py --demo

report:
	python3 ctxscope.py examples/agent-trace.json --html docs/report.html --md docs/report.md
	@echo "wrote docs/report.html and docs/report.md"

examples:
	python3 examples/make_trace.py

clean:
	rm -rf __pycache__ tests/__pycache__ examples/__pycache__ docs/report.html docs/report.md build dist *.egg-info
