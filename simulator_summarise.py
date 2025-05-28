#!/usr/bin/env python3
"""
Summarises results from a batch of vignette simulations.
Reads all YAML files from a specified simulation output directory,
extracts key metrics, and saves them to a CSV file.
"""

import argparse
import os
import yaml
import pandas as pd
from pathlib import Path
import json

def extract_data_from_file(yaml_file_path: Path, all_loaded_case_answers: dict[str, dict] | None) -> dict:
    """Extracts relevant data from a single simulation YAML file."""
    with open(yaml_file_path, 'r') as f:
        data = yaml.safe_load(f)

    num_messages = len(data.get("conversation", []))
    sim_id = data.get("simulation_id", yaml_file_path.stem)
    case_file = data.get("original_case_file", "Unknown")
    diagnosis_active = data.get("diagnosis_active", False)
    examination_active = data.get("examination_active", False)
    treatment_active = data.get("treatment_active", False)
    provider = data.get("provider", "Unknown")
    model_general = data.get("model_general", "Unknown")
    model_doctor = data.get("model_doctor", "Unknown")
    final_diagnosis_statement = data.get("final_diagnosis_statement")
    
    extracted_diagnosis_raw = data.get("extracted_diagnosis")
    extracted_diagnosis_parsed = None
    if extracted_diagnosis_raw is not None: # Explicitly check for None
        try:
            if isinstance(extracted_diagnosis_raw, dict):
                extracted_diagnosis_parsed = extracted_diagnosis_raw
            elif isinstance(extracted_diagnosis_raw, str):
                cleaned_str = extracted_diagnosis_raw.strip()
                if cleaned_str.startswith("```json"):
                    cleaned_str = cleaned_str.removeprefix("```json").strip()
                if cleaned_str.endswith("```"):
                    cleaned_str = cleaned_str.removesuffix("```").strip()
                
                # Apply the newline cleaning uniformly before attempting to parse or use as string
                final_str_value = cleaned_str.replace("\\\\n", "")

                try:
                    # Attempt to parse the cleaned string as JSON
                    extracted_diagnosis_parsed = json.loads(final_str_value)
                except json.JSONDecodeError:
                    # If JSON parsing fails, use the cleaned string itself as the value
                    extracted_diagnosis_parsed = final_str_value 
            else:
                # For types other than dict or str (e.g., list, int, float directly from YAML)
                extracted_diagnosis_parsed = extracted_diagnosis_raw
        except Exception as e: # Catch any other unexpected errors during processing
            extracted_diagnosis_parsed = {"error": f"ProcessingError: {str(e)}", "raw_value": extracted_diagnosis_raw}

    doctor_questions_count = 0
    all_history_question_ids_list = []
    doctor_questions_without_ids_val = 0

    conversation = data.get("conversation", [])
    num_conversation_entries = len(conversation)

    for i, entry in enumerate(conversation):
        if not isinstance(entry, dict):
            print(f"Warning: Skipping non-dictionary entry in conversation for {yaml_file_path.name}: {entry}")
            continue
        speaker = entry.get("speaker")
        message = entry.get("message", "")
        q_ids = entry.get("question_id")

        if speaker == "Patient":
            if q_ids:
                if isinstance(q_ids, list):
                    all_history_question_ids_list.extend(q_id for q_id in q_ids if q_id)
                elif isinstance(q_ids, str) and q_ids:
                    all_history_question_ids_list.append(q_ids)
        elif speaker == "Doctor":
            if "END_INTERVIEW" not in message:
                doctor_questions_count += 1
                if (i + 1) < num_conversation_entries:
                    next_entry = conversation[i+1]
                    if isinstance(next_entry, dict) and next_entry.get("speaker") == "Patient":
                        next_q_ids = next_entry.get("question_id")
                        if not next_q_ids:
                            doctor_questions_without_ids_val += 1
    
    unique_question_ids_covered_val = len(set(all_history_question_ids_list))
    history_question_ids_json_list_val = json.dumps(all_history_question_ids_list)

    num_checklist_items_asked = -1
    checklist_completion_rate = -1.0
    
    specific_case_answer_data = None
    case_file_from_sim = case_file # Renaming for clarity in this block

    if all_loaded_case_answers and case_file_from_sim != "Unknown":
        # Attempt 1: Direct match (e.g., case_file_from_sim is "case_definitions/caseX.yaml")
        specific_case_answer_data = all_loaded_case_answers.get(case_file_from_sim)

        # Attempt 2: If no direct match, and case_file_from_sim is an old path (e.g., "cases/caseX.yaml")
        # construct a key based on the new structure ("case_definitions/caseX.yaml") and try that.
        if not specific_case_answer_data:
            sim_path_obj = Path(case_file_from_sim)
            if len(sim_path_obj.parts) > 1 and sim_path_obj.parts[0] == "cases":
                new_structure_key = str(Path("case_definitions") / sim_path_obj.name)
                specific_case_answer_data = all_loaded_case_answers.get(new_structure_key)
                # if specific_case_answer_data:
                #     print(f"Info: Matched old path '{case_file_from_sim}' to new key '{new_structure_key}'.")

        # Attempt 3: If still no match AND case_file_from_sim is a basename (e.g., "caseX.yaml")
        # try matching against the basename of the keys in all_loaded_case_answers (which are like "case_definitions/caseY.yaml").
        if not specific_case_answer_data and (Path(case_file_from_sim).name == case_file_from_sim):
            for loaded_key_str, loaded_value in all_loaded_case_answers.items():
                if Path(loaded_key_str).name == case_file_from_sim:
                    specific_case_answer_data = loaded_value
                    # print(f"Info: Matched basename '{case_file_from_sim}' to answer data for '{loaded_key_str}'.")
                    break 

    if specific_case_answer_data and "history_questions" in specific_case_answer_data:
        checklist_items = specific_case_answer_data["history_questions"]
        total_checklist_items = len(checklist_items)
        
        if total_checklist_items > 0:
            asked_qids_set = set(all_history_question_ids_list)
            current_items_asked_count = 0
            for item_definition in checklist_items:
                item_specific_ids = set(item_definition.get('ids', []))
                if not item_specific_ids.isdisjoint(asked_qids_set):
                    current_items_asked_count += 1
            num_checklist_items_asked = current_items_asked_count
            checklist_completion_rate = num_checklist_items_asked / total_checklist_items
        else: # total_checklist_items is 0
            num_checklist_items_asked = 0
            checklist_completion_rate = 0.0 
    # Updated warning condition:
    elif not specific_case_answer_data and case_file_from_sim != "Unknown" and all_loaded_case_answers and len(all_loaded_case_answers) > 0:
         print(f"Warning: No matching answer data found for case file '{case_file_from_sim}' (tried direct, old path, and basename match). Checklist metrics will be -1.")
    
    # --- New: Extract Diagnosis Classification Details ---
    diag_classification_details = data.get("diagnosis_classification_details")
    diag_classification = None
    diag_classification_confidence = None
    diag_explanation = None

    if isinstance(diag_classification_details, dict):
        diag_classification = diag_classification_details.get("classification")
        diag_classification_confidence = diag_classification_details.get("confidence")
        diag_explanation = diag_classification_details.get("explanation")
    elif diag_classification_details: # If it exists but is not a dict (e.g. an error string)
        diag_explanation = str(diag_classification_details) # Store the raw value in explanation
        diag_classification = "Error" # Mark as error

    # --- New: Extract Treatment Details ---
    extracted_treatments_val = data.get("extracted_treatments") # Should be a list or null
    treatment_classification_details_val = data.get("treatment_classification_details")
    
    num_correct_treatments = -1
    num_palliative_treatments = -1
    num_unnecessary_harmful_treatments = -1
    treatment_classification_explanation = None
    num_not_found_treatments = -1 # Initialize new counter
    num_errored_treatments_classification = -1 # Initialize new counter for parsing errors

    if isinstance(treatment_classification_details_val, dict):
        num_correct_treatments = treatment_classification_details_val.get("correct_count", -1)
        num_palliative_treatments = treatment_classification_details_val.get("palliative_count", -1)
        num_unnecessary_harmful_treatments = treatment_classification_details_val.get("unnecessary_or_harmful_count", -1)
        treatment_classification_explanation = treatment_classification_details_val.get("explanation")
        
        classification_details_list = treatment_classification_details_val.get("classification_details")
        if isinstance(classification_details_list, list):
            current_not_found_count = 0
            current_errored_classification_count = 0 # Counter for this loop
            for detail in classification_details_list:
                if isinstance(detail, dict):
                    category = detail.get("category")
                    if category == "not found":
                        current_not_found_count += 1
                    elif category == "error_parsing_classification":
                        current_errored_classification_count += 1
            num_not_found_treatments = current_not_found_count
            num_errored_treatments_classification = current_errored_classification_count
        # raw_treatment_classification_details_json = json.dumps(treatment_classification_details_val.get("classification_details"))
    elif treatment_classification_details_val: # Error string or similar
        treatment_classification_explanation = str(treatment_classification_details_val)

    return {
        "simulation_id": sim_id,
        "provider": provider,
        "model_general": model_general,
        "model_doctor": model_doctor,
        "num_messages": num_messages,
        "case_file": case_file,
        "diagnosis_active": diagnosis_active,
        "examination_active": examination_active,
        "treatment_active": treatment_active,
        "final_diagnosis_statement": final_diagnosis_statement,
        "extracted_diagnosis": extracted_diagnosis_parsed, 
        "doctor_questions_count": doctor_questions_count,
        "all_history_question_ids": history_question_ids_json_list_val,
        "unique_question_ids_covered": unique_question_ids_covered_val,
        "doctor_questions_without_ids": doctor_questions_without_ids_val,
        "num_checklist_items_asked": num_checklist_items_asked,
        "checklist_completion_rate": checklist_completion_rate,
        "source_file": yaml_file_path.name,
        "diag_classification": diag_classification,
        "diag_classification_confidence": diag_classification_confidence,
        "diag_explanation": diag_explanation,
        "extracted_treatments": json.dumps(extracted_treatments_val) if extracted_treatments_val is not None else None, # Store as JSON string
        "num_correct_treatments": num_correct_treatments,
        "num_palliative_treatments": num_palliative_treatments,
        "num_unnecessary_harmful_treatments": num_unnecessary_harmful_treatments,
        "num_not_found_treatments": num_not_found_treatments,
        "num_errored_treatments_classification": num_errored_treatments_classification,
        "treatment_classification_explanation": treatment_classification_explanation
    }

def main():
    parser = argparse.ArgumentParser(description="Summarise vignette simulation results.")
    parser.add_argument(
        "input_dir", 
        type=str, 
        help="Path to the directory containing simulation YAML files (e.g., outputs/YYYYMMDD_HHMMSS)."
    )
    parser.add_argument(
        "--case_answer_dir",
        type=str,
        default="case_answers",
        help="Path to the directory containing case answer YAML files (e.g., case_answers/). Expected format: <original_case_name>_answer.yaml"
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    if not input_path.is_dir():
        print(f"Error: Input directory '{input_path}' not found or is not a directory.")
        return

    # Load all case answer key YAMLs
    all_loaded_case_answers = {}
    case_answer_path = Path(args.case_answer_dir)
    if case_answer_path.is_dir():
        print(f"Searching for answer files in '{case_answer_path}'...")
        found_any_answers = False
        for answer_file_path in case_answer_path.glob("*_answer.yaml"):
            found_any_answers = True
            original_case_stem = answer_file_path.stem.removesuffix("_answer")
            
            # Construct key assuming original case files are in "case_definitions/"
            # e.g., answer "case_answers/case3_answer.yaml" -> key "case_definitions/case3.yaml"
            key_path_for_answers = Path("case_definitions") / f"{original_case_stem}.yaml"

            try:
                with open(answer_file_path, 'r') as f:
                    answer_data = yaml.safe_load(f)
                if isinstance(answer_data, dict) and "history_questions" in answer_data:
                    all_loaded_case_answers[str(key_path_for_answers)] = answer_data # Use new key structure
                    print(f"  Loaded answer data for key '{key_path_for_answers}' from '{answer_file_path.name}'")
                else:
                    print(f"Warning: Answer file '{answer_file_path.name}' is malformed or missing 'history_questions'. Skipping.")
            except Exception as e:
                print(f"Error loading answer file {answer_file_path.name}: {e}. Skipping.")
        if not found_any_answers:
            print(f"No '*_answer.yaml' files found in '{case_answer_path}'. Checklist metrics may be -1.")
    else:
        print(f"Warning: Case answer directory '{case_answer_path}' not found. Checklist metrics will be -1.")

    all_data = []
    yaml_files = sorted(input_path.glob("*_sim_*.yaml"))

    if not yaml_files:
        print(f"No '*_sim_*.yaml' files found in '{input_path}'.")
        return

    print(f"Found {len(yaml_files)} simulation files in '{input_path}'.")

    for yaml_file in yaml_files:
        print(f"Processing {yaml_file.name}...")
        try:
            data_row = extract_data_from_file(yaml_file, all_loaded_case_answers)
            all_data.append(data_row)
        except Exception as e:
            print(f"Error processing file {yaml_file.name}: {e}")
            # Ensure all columns, including new ones, are present in error case
            error_entry = {
                "simulation_id": yaml_file.stem, "num_messages": -1, "case_file": "Error",
                "provider": "Error",
                "model_general": "Error",
                "model_doctor": "Error",
                "diagnosis_active": False,
                "examination_active": False,
                "treatment_active": False,
                "final_diagnosis_statement": f"Error processing file: {e}",
                "extracted_diagnosis": f"Error processing file: {e}", "doctor_questions_count": -1,
                "all_history_question_ids": "[]", "unique_question_ids_covered": -1,
                "doctor_questions_without_ids": -1, "num_checklist_items_asked": -1,
                "checklist_completion_rate": -1.0, "source_file": yaml_file.name,
                "diag_classification": "Error", 
                "diag_classification_confidence": None,
                "diag_explanation": f"Error processing file during extraction: {e}",
                "extracted_treatments": None,
                "num_correct_treatments": -1,
                "num_palliative_treatments": -1,
                "num_unnecessary_harmful_treatments": -1,
                "num_not_found_treatments": -1,
                "num_errored_treatments_classification": -1,
                "treatment_classification_explanation": f"Error processing file: {e}"
            }
            all_data.append(error_entry)

    if not all_data:
        print("No data extracted. Exiting.")
        return

    df = pd.DataFrame(all_data)
    
    columns_order = [
        "simulation_id", 
        "provider",
        "model_general",
        "model_doctor",
        "case_file", 
        "diagnosis_active", 
        "examination_active",
        "treatment_active",
        "num_messages", 
        "doctor_questions_count", 
        "all_history_question_ids", "unique_question_ids_covered", "doctor_questions_without_ids",
        "num_checklist_items_asked", "checklist_completion_rate",
        "final_diagnosis_statement", "extracted_diagnosis", 
        "diag_classification", "diag_classification_confidence", "diag_explanation",
        "extracted_treatments", 
        "num_correct_treatments", "num_palliative_treatments", "num_unnecessary_harmful_treatments", 
        "num_not_found_treatments",
        "num_errored_treatments_classification",
        "treatment_classification_explanation",
        "source_file"
    ]
    for col in columns_order:
        if col not in df.columns:
            df[col] = None # Add missing columns with None
    df = df[columns_order]

    timestamp_suffix = input_path.name
    output_filename = f"summary_{timestamp_suffix}.csv"
    output_csv_path = input_path / output_filename

    try:
        df.to_csv(output_csv_path, index=False)
        print(f"Summary saved to {output_csv_path}")
    except Exception as e:
        print(f"Error saving summary CSV to {output_csv_path}: {e}")

if __name__ == "__main__":
    main() 