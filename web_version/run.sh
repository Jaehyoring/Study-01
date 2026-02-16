# Created: 2026-02-16 22:30
#!/bin/bash
cd "$(dirname "$0")/.."
source .venv/bin/activate
python3 web_version/digit_recognizer.py
