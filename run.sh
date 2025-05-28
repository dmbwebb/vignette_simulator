#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# --- Configuration ---
NUM_SIMS=100 # Number of simulations to run FOR EACH CASE
# Define the list of case files to process
CASE_FILES_TO_RUN=(
  "case_definitions/case1.yaml"
  "case_definitions/case2.yaml"
  "case_definitions/case3.yaml"
  "case_definitions/case4.yaml"
  "case_definitions/case5.yaml"
  # "case_definitions/mrcpuk_case2.yaml"
  # Add more case files here, e.g., "case_definitions/case3.yaml"
)
# PROMPT_FILE is now determined by --prompt-dir in the python script
PROVIDER="OpenAI"
MODEL="gpt-4.1"
LANGUAGE="hi" # Language setting: "en" for English, "hi" for Hindi

# Define a list of specific models for the doctor. 
# If the list is empty or commented out, the general MODEL will be used for the doctor.
# To use the general model for some runs and a specific model for others, include "None" or an empty string in the list.
# Example: MODEL_DOCTOR_LIST=("gpt-4.1" "None" "gpt-3.5-turbo")
# MODEL_DOCTOR_LIST=("gpt-4.1" "gpt-4o-mini" "gpt-4o" "o3" "o4-mini") 
# MODEL_DOCTOR_LIST=("o3") 
MODEL_DOCTOR_LIST=("gpt-4.1") 

# Set the desired behavior using boolean flags
# --diagnosis: enables final diagnosis, extraction, classification, and uses diagnosis prompt (if --examination is false)
# --examination: enables examination prompt and implies diagnosis behavior
DIAGNOSIS_ACTIVE=true
EXAMINATION_ACTIVE=true # If true, uses doc_prompt_diagnosis_examinations.txt. If false and DIAGNOSIS_ACTIVE is true, uses doc_prompt_diagnosis.txt. If both false, uses doc_prompt.txt (summary).
TREATMENT_ACTIVE=true   # If true, enables treatment recommendation, extraction, and classification
REFERRAL_ACTIVE=false   # If true, enables referral mode

PROMPT_DIR="prompts" # Directory where prompt files are located

# --- Continue Mode Configuration ---
CONTINUE_PREVIOUS_BATCH=false # Set to true to attempt to continue the most recent batch
# If CONTINUE_PREVIOUS_BATCH is true, and a previous batch directory is found,
# the --continue-batch argument will be passed to the python script.

echo "--- Running Vignette Simulator for Multiple Cases ---"
echo "Case Files to Process:"
for f in "${CASE_FILES_TO_RUN[@]}"; do echo "  - $f"; done
echo "Number of simulations per case: $NUM_SIMS"
# echo "Mode: $MODE" # Old MODE variable removed
echo "Diagnosis Active: $DIAGNOSIS_ACTIVE"
echo "Examination Active: $EXAMINATION_ACTIVE"
echo "Treatment Active: $TREATMENT_ACTIVE"
echo "Referral Active: $REFERRAL_ACTIVE"
echo "Language: $LANGUAGE"
echo "Prompt Directory: $PROMPT_DIR"
echo "Provider: $PROVIDER"
echo "General Model: $MODEL"
# Display the list of doctor models
if [ ${#MODEL_DOCTOR_LIST[@]} -gt 0 ]; then
  echo "Doctor Models to iterate through:"
  for doc_model in "${MODEL_DOCTOR_LIST[@]}"; do
    if [[ -z "$doc_model" || "$doc_model" == "None" ]]; then
      echo "  - (using general model - $MODEL)"
    else
      echo "  - $doc_model"
    fi
  done
else
  echo "Doctor Model: (using general model - $MODEL for all runs, as MODEL_DOCTOR_LIST is empty)"
fi
echo "-----------------------------------"

if [ ${#CASE_FILES_TO_RUN[@]} -eq 0 ]; then
  echo "No case files specified in CASE_FILES_TO_RUN. Exiting."
  exit 0
fi

# --- Determine Batch Directory and Continue Argument ---
PYTHON_CONTINUE_ARG=""
BATCH_OUTPUT_DIR_TO_USE=""

if [ "$CONTINUE_PREVIOUS_BATCH" = "true" ]; then
  echo "Attempting to continue previous batch..."
  # Find the most recently modified directory in 'outputs'
  # This assumes directories in 'outputs' are timestamped or otherwise sortable by time if they are batch directories
  LATEST_BATCH_DIR=$(ls -td outputs/*/ 2>/dev/null | head -n 1)

  if [ -n "$LATEST_BATCH_DIR" ] && [ -d "$LATEST_BATCH_DIR" ]; then
    # Trim trailing slash if present (from ls fallback)
    LATEST_BATCH_DIR_TRIMMED=${LATEST_BATCH_DIR%/}
    echo "Found most recent directory: $LATEST_BATCH_DIR_TRIMMED. This will be used for --continue-batch."
    PYTHON_CONTINUE_ARG="--continue-batch \"$LATEST_BATCH_DIR_TRIMMED\""
    BATCH_OUTPUT_DIR_TO_USE=$LATEST_BATCH_DIR_TRIMMED # Used for final message if script doesn't output its own dir
  else
    echo "No previous batch directory found in 'outputs/' or 'outputs/' doesn't exist. Starting a new batch."
    # CONTINUE_PREVIOUS_BATCH remains true, but PYTHON_CONTINUE_ARG remains empty, so python script starts new batch.
    # The python script will create a new timestamped dir, and its output will be captured for OUTPUT_DIR.
  fi
else
  echo "Not in continue mode. A new batch directory will be created by the Python script."
  # PYTHON_CONTINUE_ARG remains empty
fi

# Run the simulator for all specified cases
# The python script now handles looping internally
# Output is sent to terminal live via tee to /dev/stderr, and also captured to FULL_COMMAND_OUTPUT
python_command="python3 -u vignette_simulator.py \
  --cases$(printf " \"%s\"" "${CASE_FILES_TO_RUN[@]}") \
  --prompt-dir \"$PROMPT_DIR\" \
  --provider \"$PROVIDER\" \
  --model \"$MODEL\" \
  --n_sims \"$NUM_SIMS\" \
  --language \"$LANGUAGE\""

# Add the continue argument if it was set
if [ -n "$PYTHON_CONTINUE_ARG" ]; then
  python_command+=" $PYTHON_CONTINUE_ARG"
fi

# Add diagnosis and examination flags if active
if [ "$DIAGNOSIS_ACTIVE" = "true" ]; then
  python_command+=" \
  --diagnosis"
fi

if [ "$EXAMINATION_ACTIVE" = "true" ]; then
  python_command+=" \
  --examination"
fi

# Add treatment flag if active
if [ "$TREATMENT_ACTIVE" = "true" ]; then
  python_command+=" \
  --treatment"
fi

# Add referral flag if active
if [ "$REFERRAL_ACTIVE" = "true" ]; then
  python_command+=" \
  --referral"
fi

# Add model-doctors only if MODEL_DOCTOR_LIST is not empty
if [ ${#MODEL_DOCTOR_LIST[@]} -gt 0 ]; then
  # Construct the --model-doctors argument string
  model_doctor_args=""
  for doc_model in "${MODEL_DOCTOR_LIST[@]}"; do
    # Pass "None" as a string if doc_model is empty or literally "None"
    if [[ -z "$doc_model" || "$doc_model" == "None" ]]; then
      model_doctor_args+=" \"None\"" # Pass the string "None"
    else
      model_doctor_args+=" \"$doc_model\""
    fi
  done
  python_command+=" \
  --model-doctors$model_doctor_args"
fi

# Output is sent to terminal live via tee to /dev/stderr, and also captured to FULL_COMMAND_OUTPUT
FULL_COMMAND_OUTPUT=$(eval $python_command 2>&1 | tee /dev/stderr)

# Check the exit code of the python script (first command in the pipeline, which is eval)
SIMULATOR_EXIT_CODE=${PIPESTATUS[0]}

if [ $SIMULATOR_EXIT_CODE -ne 0 ]; then
  echo "-----------------------------------"
  echo "ERROR: Vignette simulator failed with exit code $SIMULATOR_EXIT_CODE."
  echo "Full output was displayed above."
  echo "-----------------------------------"
  # TMP_OUTPUT will be cleaned up by the trap
  exit 1
fi

# Extract the output directory from the temp file
# The Python script now prints "Results saved in: <directory_path>"
OUTPUT_DIR=$(echo "$FULL_COMMAND_OUTPUT" | grep "Results saved in:" | sed 's/Results saved in: //')

# Trim potential leading/trailing whitespace
OUTPUT_DIR=$(echo "$OUTPUT_DIR" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

# Check if OUTPUT_DIR was extracted
if [ -z "$OUTPUT_DIR" ]; then
  echo "-----------------------------------"
  echo "ERROR: Could not determine output directory from simulator output (displayed above)."
  echo "Attempting fallback: finding the latest directory in 'outputs'..."
  # Fallback: Find the latest modified directory in 'outputs'
  LATEST_DIR=$(ls -td outputs/*/ 2>/dev/null | head -n 1)
  if [ -n "$LATEST_DIR" ] && [ -d "$LATEST_DIR" ]; then
     # Trim trailing slash if present (from ls fallback)
     OUTPUT_DIR=${LATEST_DIR%/}
     echo "Using latest directory: $OUTPUT_DIR (Warning: This may not be the correct one if multiple batches ran recently)"
  else
     echo "ERROR: Fallback failed. Could not find a suitable output directory. Exiting."
     echo "Full simulator output was displayed above."
     echo "-----------------------------------"
     exit 1
  fi
fi

echo "-----------------------------------"
echo "Output directory identified as: $OUTPUT_DIR"

# Check if the directory actually exists
if [ ! -d "$OUTPUT_DIR" ]; then
  echo "ERROR: Identified output directory '$OUTPUT_DIR' does not exist or is not a directory."
  echo "Full output from simulator was displayed above."
  echo "-----------------------------------"
  exit 1
fi

echo "--- Running summarizer on batch output: $OUTPUT_DIR ---"

# Run the summarizer script on the batch output directory
python3 simulator_summarise.py "$OUTPUT_DIR" --case_answer_dir case_answers

# Check the exit code of the summarizer script
SUMMARIZER_EXIT_CODE=$?
if [ $SUMMARIZER_EXIT_CODE -eq 0 ]; then
  echo "Summarizer finished successfully."
else
  echo "ERROR: Summarizer script failed with exit code $SUMMARIZER_EXIT_CODE."
  echo "-----------------------------------"
  exit 1
fi

# Trap will automatically clean up $TMP_OUTPUT on successful exit here

echo "-----------------------------------"
echo "Script finished successfully."
echo "Output directory: $OUTPUT_DIR"
echo "-----------------------------------"