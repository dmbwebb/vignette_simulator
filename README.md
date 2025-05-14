# Vignette Simulator: Evaluating LLMs in Medical Dialogue

## Purpose

The Vignette Simulator is a research tool designed to rigorously evaluate the capabilities of Large Language Models (LLMs) in a simulated medical consultation setting. Its primary goals are:

*   **Assess Medical Knowledge Encoding**: To test how accurately LLMs can embody the role of a physician in gathering a comprehensive patient history.
*   **Evaluate Diagnostic Reasoning**: To measure an LLM's ability to formulate a differential diagnosis based on the information collected during the simulated patient interview.
*   **Standardized Evaluation Framework**: To provide a consistent and reproducible method for benchmarking different LLMs and prompting strategies in the context of clinical dialogue.

This tool allows researchers and developers to systematically analyze LLM performance in tasks critical to potential healthcare applications, focusing on structured information gathering and clinical reasoning.

## Overview

The Vignette Simulator orchestrates automated, text-based conversations between an LLM-powered "Doctor Simulator" and a pre-scripted "Patient Simulator." This process generates rich datasets of simulated clinical encounters.

1.  **Doctor Simulator (`vignette_simulator.py`)**:
    *   **Role**: Acts as the LLM-driven physician.
    *   **Behavior**: Follows a user-defined prompt (e.g., to gather a history, or to gather a history and then provide a diagnosis). It dynamically asks questions based on the evolving conversation.
    *   **Control**: Its operational mode (`"summary"` for information gathering, or `"diagnosis"` for information gathering plus diagnostic reasoning) is configurable.

2.  **Patient Simulator (integrated within `vignette_simulator.py`)**:
    *   **Role**: Provides responses as the patient.
    *   **Behavior**: Draws answers from a structured "Case Definition" YAML file. It uses semantic similarity to match the Doctor Simulator's questions to the closest predefined question in its database and delivers the corresponding answer. This ensures consistent patient information across multiple simulations and LLM tests.

3.  **Simulation Workflow & Analysis**:
    *   **Input**: Case Definition files (patient scripts) and Doctor Prompts.
    *   **Process**: The Doctor and Patient simulators interact, generating a full dialogue transcript.
    *   **Output**: Each simulation run produces a detailed YAML file containing the conversation, metadata, and (if in diagnosis mode) the LLM's diagnosis.
    *   **Summarization**: A dedicated **Summarizer Script (`simulator_summarise.py`)** processes batches of these simulation outputs. It extracts key metrics, compares dialogue content against predefined checklists (from "Case Answer Key" files), and evaluates the accuracy of generated diagnoses. The results are compiled into a comprehensive CSV file for easy analysis.

This system enables the detailed study of LLM questioning strategies, information extraction capabilities, and diagnostic accuracy in a controlled environment.

## Features

The Vignette Simulator offers a range of features for comprehensive LLM evaluation in simulated medical dialogues:

**Core Simulation Engine (`vignette_simulator.py`):**

*   **Autonomous Dialogues**: Conducts complete, automated doctor-patient conversations without requiring user input during the simulation.
*   **LLM Flexibility**: Supports multiple LLM providers (e.g., OpenAI, Anthropic) and various models, allowing for comparative studies.
*   **Configurable Modes**: Operates in different modes:
    *   `"summary"` mode: Focuses on the LLM's ability to gather comprehensive information, automatically detecting when sufficient history is likely collected.
    *   `"diagnosis"` mode: Extends the `"summary"` mode by prompting the LLM to provide a diagnosis and rationale after information gathering.
*   **Consistent Patient Behavior**: Utilizes case definition files to ensure the patient simulator provides standardized responses, crucial for reproducible experiments.
*   **Dynamic Question Matching**: Employs semantic matching to link the doctor's free-text questions to the closest available question-answer pair in the case definition, allowing for natural conversation flow.
*   **Detailed Output**: Generates a full dialogue transcript for each simulation, including per-turn metadata, raw LLM messages, and any generated diagnosis, all saved in structured YAML files.

**Batch Analysis & Evaluation (`simulator_summarise.py`):**

*   **Automated Metrics Extraction**: Processes multiple simulation YAML outputs to extract key performance indicators.
*   **Checklist-Based Evaluation**: Compares the doctor LLM's questions against predefined checklists in "Case Answer Key" files to quantify information-gathering completeness.
*   **Diagnosis Accuracy Assessment**: Scores the LLM's generated diagnoses against ground truth answers defined in the answer keys.
*   **Aggregated Reporting**: Summarizes results from numerous simulations into a single CSV file, facilitating large-scale analysis and comparison of different models, prompts, or cases.

**Usability & Customization:**

*   **Streamlined Execution**: Includes a `run.sh` script to easily configure and execute batches of simulations and subsequent summarization.
*   **Customizable Prompts**: Allows users to define and modify doctor LLM prompts in simple text files to steer behavior and test different interaction styles.
*   **Expandable Case Library**: Enables users to create new patient case definitions and corresponding answer keys in YAML format to broaden the scope of evaluations.

## Setup

### Using the setup script

1. Ensure you have Python 3.6+ installed
2. Run the setup script to create a virtual environment and install dependencies:
   ```
   chmod +x setup.sh
   ./setup.sh
   ```
3. Activate the virtual environment:
   ```
   source venv/bin/activate
   ```
4. **Directory Structure**: The project uses the following key directories:
    * `case_definitions/`: Contains YAML files defining the patient cases (questions, answers, metadata).
    * `case_answers/`: Contains YAML files with answer keys for each case, used by the summarizer for metrics like checklist completion and diagnosis accuracy.
    * `prompts/`: Contains text files for different doctor LLM prompts (e.g., for summary mode, diagnosis mode).
    * `outputs/`: Default directory where simulation results (individual YAMLs and summary CSVs) are saved.
        * `outputs/curated_outputs/`: Contains YAML outputs from key/main simulation runs that are often referenced.
    * `figs/`: Contains graphs and figures visualizing the main results of simulations.

### Manual setup

1. Ensure you have Python 3.6+ installed
2. Create a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install required dependencies:
   ```
   pip install pyyaml requests python-dotenv pandas
   ```
4. Configure your API keys in one of two ways:

   **Option A**: Using environment variables in a `.env` file:
   ```
   # Create a .env file from the example
   cp .env.example .env

   # Edit the .env file and add your API keys
   ```

   **Option B**: Using the `api_keys.yaml` file:
   ```yaml
   api_keys:
     - name: "OpenAI"
       key: "your-openai-api-key"
     - name: "Anthropic"
       key: "your-anthropic-api-key"
   ```

   The simulator will check for API keys in both locations, with environment variables taking precedence.

## Usage

The primary way to run simulations is using the `run.sh` script, which handles running one or more simulations for specified cases and then automatically summarizing the results.

### Using `run.sh`

1. **Configure `run.sh`**:
    Open `run.sh` and modify the following variables at the top of the script:
    * `NUM_SIMS`: Number of simulations to run *for each* case file.
    * `CASE_FILES_TO_RUN`: An array of paths to the case definition files you want to process (e.g., `"case_definitions/case1.yaml"`).
    * `PROMPT_FILE`: Path to the doctor prompt file to use (e.g., `"prompts/doc_prompt_diagnosis.txt"`).
    * `PROVIDER`: LLM provider (e.g., `"OpenAI"`).
    * `MODEL`: Specific model name (e.g., `"gpt-4o-mini"`).
    * `MODE`: Simulation mode (`"summary"` or `"diagnosis"`). This should align with the chosen `PROMPT_FILE`.

2. **Execute the script**:
    ```bash
    ./run.sh
    ```
    The script will run `vignette_simulator.py` for all specified cases and then `simulator_summarise.py` on the generated output directory. Results, including individual simulation YAMLs and a summary CSV, will be saved in a timestamped subdirectory within `outputs/`.

### Running `vignette_simulator.py` directly

You can also run the simulator script directly:
```bash
python vignette_simulator.py --cases <path_to_case1.yaml> [<path_to_case2.yaml> ...] --doc-prompt <path_to_prompt.txt> --provider <ProviderName> --model <ModelName> --mode <ModeName> --n_sims <NumberOfSimulations>
```

### Running `simulator_summarise.py` directly

To summarize existing simulation results:
```bash
python simulator_summarise.py <path_to_simulation_output_directory> --case_answer_dir <path_to_case_answers_directory>
```

### Command Line Arguments

#### `vignette_simulator.py`

- `--cases`: Space-separated paths to one or more case definition files (e.g., `case_definitions/case1.yaml case_definitions/case2.yaml`). Replaces the old `--case` argument.
- `--doc-prompt`: Path to the doctor prompt file (default: `prompts/doc_prompt_diagnosis.txt`).
- `--provider`: LLM provider to use (`OpenAI` or `Anthropic`, default: `OpenAI`).
- `--model`: Model to use (default: `gpt-4o-mini`).
- `--mode`: Simulation mode (`summary` or `diagnosis`, default: `diagnosis`).
- `--n_sims`: Number of simulations to run for each case file (default: `1`).
- `--output_dir_prefix`: Optional prefix for the output directory name (default: `simulation_results`). The script will create a timestamped directory like `outputs/simulation_results_YYYYMMDD_HHMMSS/`.

*Note: The old `--output` argument for a single file path is removed. The script now always outputs to a dedicated directory.*

#### `simulator_summarise.py`

- `input_dir`: Path to the directory containing simulation YAML files from `vignette_simulator.py` (e.g., `outputs/simulation_results_YYYYMMDD_HHMMSS`). This is a positional argument.
- `--case_answer_dir`: Path to the directory containing case answer key YAML files (default: `case_answers/`). These files are used for checklist completion and diagnosis scoring.

## Case File Formats

The simulation relies on two types of YAML files: Case Definition files and Case Answer Key files.

### 1. Case Definition Files

Located in `case_definitions/` (e.g., `case_definitions/case1.yaml`). These files define the patient's persona and how they respond to questions.

- `introduction`: An introductory statement from the patient (in English and Hindi).
- `history_questions`: General history questions with their expected answers.
- `examination_questions`: Physical examination questions with their expected answers.
- `provider_treatment_questions`: Questions about treatment plans with their expected answers. (This section might be deprecated or less used with the new diagnosis mode, verify if still relevant).

Each question entry within these lists contains:
- `id`: Question identifier (e.g., `H1`, `E5`).
- `type`: Category of the question (e.g., `History`, `Examination`).
- `question_en`: Question text in English.
- `question_hi`: Question text in Hindi (optional).
- `answer_en`: Answer text in English.
- `answer_hi`: Answer text in Hindi (optional).

### 2. Case Answer Key Files

Located in `case_answers/` (e.g., `case_answers/case1_answer.yaml`). These files provide the "ground truth" for evaluating simulation outputs, used by `simulator_summarise.py`. The filename should correspond to a case definition file (e.g., `case1_answer.yaml` for `case_definitions/case1.yaml`).

Each answer key file can contain:
- `history_questions`: A list of checklist items. Each item has:
    - `item`: A descriptive name for the checklist category (e.g., "qualities of stool").
    - `ids`: A list of question `id`s (from the corresponding case definition file) that fall under this item.
- `examinations`: (Structure TBD or can be left empty if not used for specific metrics).
- `diagnosis`:
    - `correct`: A list of correct diagnostic terms.
    - `incorrect`: A list of incorrect diagnostic terms.
- `treatments`:
    - `correct`: A list of correct treatments.
    - `palliative`: A list of palliative treatments.
    - `unnecessary_or_harmful`: A list of unnecessary or harmful treatments.

## Output Format

### Simulation Output (from `vignette_simulator.py`)

The `vignette_simulator.py` script creates a timestamped subdirectory within `outputs/` (e.g., `outputs/simulation_results_YYYYMMDD_HHMMSS/`). Inside this directory, it saves one YAML file per simulation run (e.g., `case1_sim_1.yaml`, `case1_sim_2.yaml`).

Each simulation YAML file contains:
- `simulation_id`: Unique identifier for the simulation (e.g., `case1_sim_1`).
- `original_case_file`: Path to the case definition file used.
- `prompt_file`: Path to the prompt file used.
- `provider`: LLM provider used.
- `model`: LLM model used.
- `mode`: Simulation mode (`summary` or `diagnosis`).
- `conversation`: A list of conversation steps, each with:
  - `step`: Step number.
  - `speaker`: "Doctor" or "Patient".
  - `question_id`: The ID(s) of the matched question from the case definition file (for patient responses). Can be a list if multiple questions were matched.
  - `message`: The actual text content.
- `raw_messages`: The raw message objects exchanged with the LLM.
- `final_diagnosis_statement`: The complete final statement from the doctor if in "diagnosis" mode.
- `extracted_diagnosis`: A JSON string or object containing a structured version of the diagnosis (if applicable).
- `diagnosis_classification_details`: A dictionary containing:
    - `classification`: The classified diagnosis (e.g., "Correct", "Partially Correct", "Incorrect").
    - `confidence`: A confidence score/statement for the classification.
    - `explanation`: An explanation for the classification.

### Summary Output (from `simulator_summarise.py`)

The `simulator_summarise.py` script processes the simulation YAML files in an output directory and generates a CSV file (e.g., `summary_simulation_results_YYYYMMDD_HHMMSS.csv`) in that same directory.

The CSV file includes columns such as:
- `simulation_id`
- `num_messages`
- `case_file`
- `mode`
- `doctor_questions_count`
- `all_history_question_ids` (JSON list of all question IDs covered by patient)
- `unique_question_ids_covered`
- `doctor_questions_without_ids`
- `num_checklist_items_asked` (based on `case_answers/` files)
- `checklist_completion_rate` (based on `case_answers/` files)
- `final_diagnosis_statement`
- `extracted_diagnosis` (parsed from the simulation YAML)
- `diag_classification`
- `diag_classification_confidence`
- `diag_explanation`
- `source_file` (name of the input simulation YAML file)

## Customization

- Edit the doctor prompts in the `prompts/` directory to change the doctor's behavior for different modes.
- Create new case definition files (YAML) in `case_definitions/` following the format of the existing ones.
- For new cases, create corresponding case answer key files (YAML) in `case_answers/` if you want to utilize the checklist completion and diagnosis accuracy metrics from the `simulator_summarise.py` script.
- Modify the patient simulator's matching threshold (currently 0.6) in `PatientSimulator._find_best_match` within `vignette_simulator.py`.