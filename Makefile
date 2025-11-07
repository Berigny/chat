PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install run

install:
	$(PIP) install -r requirements.txt

run:
	streamlit run streamlit_app.py
