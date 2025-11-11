PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install run chat demo

install:
	$(PIP) install -r requirements.txt

run:
	streamlit run streamlit_app.py

chat:
	streamlit run chat_app.py

demo:
	streamlit run chat_demo_app.py
