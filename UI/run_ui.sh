#!/bin/bash
# TrackCoach Web UI Launch Script (Linux/Mac)

echo "========================================"
echo "TrackCoach Web UI Launch Script"
echo "========================================"
echo ""

cd "$(dirname "$0")"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python not found, please install Python 3.7+"
    exit 1
fi

# Check if streamlit is installed
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "Detected streamlit is not installed, installing..."
    pip3 install streamlit>=1.28.0
    if [ $? -ne 0 ]; then
        echo "Error: streamlit installation failed"
        exit 1
    fi
fi

echo "Starting Streamlit application..."
echo ""
echo "Browser will open automatically, if not, please visit: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 -m streamlit run app.py
