.PHONY: install test demo clean

install:
	python -m pip install -e .

test:
	python -m pytest -q

demo:
	python examples/demo.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache build dist *.egg-info
