#!/bin/bash
# Run the Streamlit demo for Polars Expression Transformer

# Check if Streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit is not installed. Installing demo dependencies..."
    poetry install --with demo
fi

# Run the Streamlit app
echo "Starting Polars Expression Transformer Demo..."
streamlit run demo/streamlit_app.py