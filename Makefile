install:
	pip --update pip &&\
		pip install -r requirements.txt

test:
	python -m pytest -vv --cov=main --cov=mylib test_*.py

format:
	black *.py

lint:
	pylint --disable=R,C --ignore-patterns=test_.*?py *.py mylib/*.py
