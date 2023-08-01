
requirements/requirements.txt: requirements/requirements.in
	pip-compile --output-file requirements/requirements.txt requirements/requirements.in


venv: requirements/requirements.in
	python3 -m venv venv
	./venv/bin/pip install -r requirements/requirements.txt
	./venv/bin/python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
	touch venv
