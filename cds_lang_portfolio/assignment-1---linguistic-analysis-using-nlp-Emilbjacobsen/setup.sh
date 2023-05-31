#!/usr/bin/env bash

#making virtual environment
python3 -m venv assignment1_lang_env

source ./assignment1_lang_env/bin/activate

#upgrading pip and installing requirements
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m spacy download en_core_web_sm






#deactivate env
#deactivate