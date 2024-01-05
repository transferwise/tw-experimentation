package_name = tw_experimentation
coverage_target = 70
max_line_length = 88

venv_name = venv
venv_activate_path := ./$(venv_name)/bin/activate
cov_args := --cov $(package_name) --cov-fail-under=$(coverage_target) --cov-report=term-missing
not_slow = -m "not slow"

.PHONY: clean venv update lint test slowtest cov slowcov

clean:
	rm -rf ./$(venv_name)

venv:
	python3 -m venv $(venv_name) ;\
	. $(venv_activate_path) ;\
	pip install --upgrade pip setuptools wheel ;\
	pip install --upgrade -r envs/requirements-dev.txt ;\
	pip install --upgrade -r envs/requirements.txt


venvdev:
	python3 -m venv $(venv_name) ;\
	. $(venv_activate_path) ;\
	pip install --upgrade pip setuptools wheel ;\
	pip install --upgrade -r envs/requirements-dev.txt ;\

update:
	. $(venv_activate_path) ;\
	pip install --upgrade -r envs/requirements-dev.txt ;\
	pip install --upgrade -r envs/requirements.txt

lint:
	. $(venv_activate_path) ;\
	flake8 --max-line-length=$(max_line_length)

test:
	. $(venv_activate_path) ;\
	py.test $(not_slow) --disable-warnings

slowtest:
	. $(venv_activate_path) ;\
	py.test

cov:
	. $(venv_activate_path) ;\
	py.test $(cov_args) $(not_slow)

slowcov:
	. $(venv_activate_path) ;\
	py.test $(cov_args)

streamlit-run:
	streamlit run tw_experimentation/streamlit/Main.py

run-streamlit-poetry:
	poetry run streamlit run tw_experimentation/streamlit/Main.py

set-up-poetry-mac:
	pip install poetry ;\
	poetry config virtualenvs.in-project true ;\
	brew install pyenv ;\
	pyenv install --list | grep " 3\.[9]" ;\
	poetry install

set-up-poetry-windows:
	pip install poetry ;\
	poetry config virtualenvs.in-project true ;\
	pip install pyenv-win ;\
	pyenv-win install --list | grep " 3\.[9]" ;\
	poetry install