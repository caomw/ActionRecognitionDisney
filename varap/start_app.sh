#!/usr/bin/env bash
# specify flask application script
export FLASK_APP=varap_service.py

# enable debug mode so the server would reload itself on code changes
export FLASK_DEBUG=1

# start the application
python -m flask run --host=0.0.0.0

