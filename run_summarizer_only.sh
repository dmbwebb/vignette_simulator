#!/bin/bash

echo "--- Running Summarizer on Most Recent Output Directory ---"

# Find the latest modified directory in 'outputs'
LATEST_DIR=$(ls -td outputs/*/ 2>/dev/null | head -n 1)

# Check if LATEST_DIR was found
if [ -z "$LATEST_DIR" ]; then
  echo "ERROR: Could not find any subdirectories in 'outputs/'."
  exit 1
fi

# Trim trailing slash if present (from ls -d)
OUTPUT_DIR=${LATEST_DIR%/}

echo "Most recent output directory found: $OUTPUT_DIR"

# Check if the directory actually exists
if [ ! -d "$OUTPUT_DIR" ]; then
  echo "ERROR: Identified directory '$OUTPUT_DIR' does not exist or is not a directory."
  exit 1
fi

echo "--- Running simulator_summarise.py on: $OUTPUT_DIR ---"

# Run the summarizer script
python3 simulator_summarise.py "$OUTPUT_DIR" --case_answer_dir case_answers

# Check the exit code of the summarizer script
SUMMARIZER_EXIT_CODE=$?
if [ $SUMMARIZER_EXIT_CODE -eq 0 ]; then
  echo "Summarizer finished successfully for $OUTPUT_DIR."
else
  echo "ERROR: Summarizer script failed with exit code $SUMMARIZER_EXIT_CODE for $OUTPUT_DIR."
  exit 1
fi

echo "-----------------------------------"
echo "Script finished successfully."
echo "Summary for directory: $OUTPUT_DIR"
echo "-----------------------------------" 