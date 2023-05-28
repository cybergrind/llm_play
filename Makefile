
requirements.txt: requirements.in
	pip-compile --output-file requirements.txt requirements.in


venv: requirements.txt
	python3 -m venv venv
	./venv/bin/pip install -r requirements.txt
	./venv/bin/pip install -e .
	touch venv
