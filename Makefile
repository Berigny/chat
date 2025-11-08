PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install run chat

install:
	$(PIP) install -r requirements.txt

run:
	streamlit run streamlit_app.py

chat:
	streamlit run chat_app.py
