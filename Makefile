
requirements/requirements.txt: requirements/requirements.in
	pip-compile --output-file requirements/requirements.txt requirements/requirements.in


venv: requirements.txt
	python3 -m venv venv
	./venv/bin/pip install -r requirements/requirements.txt
	touch venv
