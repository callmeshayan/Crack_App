#!/bin/bash
# Run image inference workflow with Python 3.11
cd "$(dirname "$0")"
/opt/homebrew/bin/python3.11 infer_image_workflow.py "$@"
