PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install run chat demo

install:
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) -m streamlit run chat_demo_app.py

chat:
	$(PYTHON) -m streamlit run admin_app.py

demo:
	$(PYTHON) -m streamlit run chat_demo_app.py
