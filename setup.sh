#!/bin/bash
# Setup script for the vignette simulator

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate


# Install required packages
# echo "Installing required packages..."`
# pip install pyyaml requests python-dotenv
pip install pyyaml requests python-dotenv pandas

echo "Setup complete! You can now run the simulator with:"
echo "source venv/bin/activate"
echo "python vignette_simulator.py"