#!/bin/bash

pip install -U -r requirements.txt

huggingface-cli login
accelerate config
