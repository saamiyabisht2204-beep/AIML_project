#!/usr/bin/env bash
# Quick startup script for Student Stress Level Predictor

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "Starting Student Stress Predictor on http://localhost:8501 ..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
