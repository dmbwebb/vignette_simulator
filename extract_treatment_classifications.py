#!/usr/bin/env python3
"""
Extracts treatment classification details from simulation YAML files.

This script scans a directory (and its subdirectories) for YAML files,
extracts simulation identifiers and treatment-specific classification data,
and compiles it into a single CSV file.
"""

import yaml
import pandas as pd
from pathlib import Path
import argparse

def extract_info_from_yaml(yaml_file_path: Path) -> list[dict]:
    """
    Extracts simulation identifiers and treatment classification details
    from a single YAML file.

    Args:
        yaml_file_path: Path to the input YAML file.

    Returns:
        A list of dictionaries, where each dictionary represents a row
        (one per treatment item, or one row with N/A for treatment fields
        if no treatment items are found).
    """
    rows = []
    try:
        with open(yaml_file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML file {yaml_file_path.name}: {e}")
        # Return a single row with error info for this file if it can't be parsed
        return [{
            "simulation_id": yaml_file_path.stem,
            "provider": "ErrorLoadingYAML",
            "model_general": "ErrorLoadingYAML",
            "model_doctor": "ErrorLoadingYAML",
            "case_file": "ErrorLoadingYAML",
            "stated_treatment": None,
            "category": None,
            "matched_key_item": None,
            "source_file_error": str(e)
        }]

    # Extract simulation identifiers
    sim_id = data.get("simulation_id", yaml_file_path.stem)
    provider = data.get("provider")
    model_general = data.get("model_general")
    model_doctor = data.get("model_doctor")
    # 'case_file' in simulator_summarise.py comes from 'original_case_file'
    case_file = data.get("original_case_file") 

    base_row_data = {
        "simulation_id": sim_id,
        "provider": provider,
        "model_general": model_general,
        "model_doctor": model_doctor,
        "case_file": case_file,
        "source_file": yaml_file_path.name # Adding source file for traceability
    }

    treatment_classification_details = data.get("treatment_classification_details")

    if isinstance(treatment_classification_details, dict):
        classification_list = treatment_classification_details.get("classification_details")
        if isinstance(classification_list, list) and classification_list:
            for item in classification_list:
                if isinstance(item, dict):
                    row = base_row_data.copy()
                    row["stated_treatment"] = item.get("stated_treatment")
                    row["category"] = item.get("category")
                    row["matched_key_item"] = item.get("matched_key_item")
                    rows.append(row)
                else:
                    # Handle malformed item in classification_list
                    row = base_row_data.copy()
                    row["stated_treatment"] = "MalformedItem"
                    row["category"] = "MalformedItem"
                    row["matched_key_item"] = "MalformedItem"
                    rows.append(row)
        else:
            # treatment_classification_details exists, but classification_details is empty or not a list
            row = base_row_data.copy()
            row["stated_treatment"] = None
            row["category"] = None
            row["matched_key_item"] = None
            rows.append(row)
    else:
        # No treatment_classification_details section or it's not a dict
        row = base_row_data.copy()
        row["stated_treatment"] = None
        row["category"] = None
        row["matched_key_item"] = None
        rows.append(row)
    
    return rows

def main():
    parser = argparse.ArgumentParser(description="Extract treatment classification details from simulation YAMLs.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="outputs/curated_outputs/curated_outputs_disagg",
        help="Directory containing simulation YAML files (recursive search)."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="outputs/curated_outputs/treatment_classifications.csv",
        help="Path to save the output CSV file."
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_csv_path = Path(args.output_csv)

    if not input_path.is_dir():
        print(f"Error: Input directory '{input_path}' not found or is not a directory.")
        return

    all_treatment_rows = []
    yaml_files = sorted(list(input_path.rglob("*.yaml"))) # rglob for recursive

    if not yaml_files:
        print(f"No '*.yaml' files found in '{input_path}' (recursively).")
        return

    print(f"Found {len(yaml_files)} YAML files in '{input_path}' and its subdirectories.")

    for yaml_file in yaml_files:
        # print(f"Processing {yaml_file.name}...") # Can be verbose for many files
        try:
            file_rows = extract_info_from_yaml(yaml_file)
            all_treatment_rows.extend(file_rows)
        except Exception as e:
            print(f"Critical error processing file {yaml_file.name}: {e}")
            # Add a placeholder row for critical errors during extraction logic
            all_treatment_rows.append({
                "simulation_id": yaml_file.stem,
                "provider": "CriticalError",
                "model_general": "CriticalError",
                "model_doctor": "CriticalError",
                "case_file": "CriticalError",
                "stated_treatment": None,
                "category": None,
                "matched_key_item": None,
                "source_file": yaml_file.name,
                "source_file_error": f"Critical processing error: {e}"
            })
            
    if not all_treatment_rows:
        print("No data extracted. Exiting.")
        return

    df = pd.DataFrame(all_treatment_rows)

    # Define column order
    columns_order = [
        "simulation_id",
        "provider",
        "model_general",
        "model_doctor",
        "case_file",
        "stated_treatment",
        "category",
        "matched_key_item",
        "source_file",
        "source_file_error" # Optional: if you want to track files with issues
    ]
    
    # Ensure all defined columns exist, add if not (e.g. source_file_error might not always be there)
    for col in columns_order:
        if col not in df.columns:
            df[col] = None 
            
    df = df[columns_order] # Reorder and select columns

    # Ensure output directory exists
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"Successfully extracted data for {len(df)} treatment entries from {len(yaml_files)} files.")
        print(f"Output saved to {output_csv_path}")
    except Exception as e:
        print(f"Error saving CSV to {output_csv_path}: {e}")

if __name__ == "__main__":
    main() 