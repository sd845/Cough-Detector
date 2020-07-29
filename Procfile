#!/bin/bash
apt update && apt install -y libglib2.0-0
gunicorn --bind=0.0.0.8000 --timeout 600 app:app
