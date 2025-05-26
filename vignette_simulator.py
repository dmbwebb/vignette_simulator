#!/usr/bin/env python3
"""
Vignette Simulator - Creates simulated conversations between a doctor and a patient using LLMs.
"""

import os
import sys
import json
import yaml
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import requests
from difflib import SequenceMatcher
from dotenv import load_dotenv
import re
from datetime import datetime, timezone
import multiprocessing

# ==== Environment Variables Management ====

def load_api_keys(api_key_file: str = "api_keys.yaml") -> Dict[str, str]:
    """Load API keys from the YAML file and environment variables"""
    # First try to load from .env file
    load_dotenv()

    # Check environment variables first
    keys = {
        "OpenAI": os.environ.get("OPENAI_API_KEY", ""),
        "Anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
    }

    # If keys are not in environment variables, try loading from YAML file
    if not keys["OpenAI"] or not keys["Anthropic"]:
        try:
            with open(api_key_file, 'r') as f:
                data = yaml.safe_load(f)

            for item in data.get('api_keys', []):
                if item['name'] in keys and not keys[item['name']]:
                    keys[item['name']] = item['key']

        except Exception as e:
            print(f"Note: Could not load API keys from {api_key_file}: {e}")
            print("Will try to use environment variables instead.")

    return keys

# ==== LLM Providers ====

class LLMProvider:
    """Base class for LLM providers"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.system_prompt = ""
        self.verbose = False  # Initialize verbose flag
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt"""
        self.system_prompt = prompt
    
    def set_verbose(self, verbose: bool):
        """Set the verbosity"""
        self.verbose = verbose
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send messages to the LLM and get a response"""
        pass

# ==== Helper function to create provider instances ====
def _create_llm_provider(provider_name: str, model_name: str, api_key: str, verbose: bool) -> LLMProvider:
    """Helper function to create and configure an LLM provider instance."""
    if provider_name == "Anthropic":
        provider = AnthropicProvider(api_key)
    else:  # Default to OpenAI
        provider = OpenAIProvider(api_key)
    provider.set_model(model_name)
    provider.set_verbose(verbose)
    return provider

class AnthropicProvider(LLMProvider):
    """Provider for Anthropic Claude API"""
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.model = "claude-3-sonnet-20240229"  # Default model
        self.api_url = "https://api.anthropic.com/v1/messages"
    
    def set_model(self, model: str):
        """Set the model to use"""
        self.model = model
        print(f"Using Anthropic model: {model}")
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send messages to Claude and get a response"""
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        # Prepare the messages with system prompt
        formatted_messages = messages.copy()
        
        data = {
            "model": self.model,
            "messages": formatted_messages,
            "system": self.system_prompt,
            "max_tokens": 4000
        }
        
        # Verbose output before API call
        if self.verbose:
            print("--- Anthropic API Call ---")
            print(f"URL: {self.api_url}")
            print(f"Model: {self.model}")
            print(f"Payload: {json.dumps(data, indent=2)}")
            print("-------------------------")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            content = result["content"][0]["text"]
            return content
        except Exception as e:
            print(f"Error communicating with Anthropic API: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            return f"Error: {str(e)}"

class OpenAIProvider(LLMProvider):
    """Provider for OpenAI GPT API"""
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.model = "gpt-4o-mini"  # Default model
        self.api_url = "https://api.openai.com/v1/chat/completions"
    
    def set_model(self, model: str):
        """Set the model to use"""
        self.model = model
        print(f"Using OpenAI model: {model}")
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send messages to GPT and get a response"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Add system prompt if it exists
        formatted_messages = []
        if self.system_prompt:
            formatted_messages.append({"role": "system", "content": self.system_prompt})
        
        # Add the rest of the messages
        formatted_messages.extend(messages)
        
        data = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": 4000
        }
        
        # Verbose output before API call
        if self.verbose:
            print("--- OpenAI API Call ---")
            print(f"URL: {self.api_url}")
            print(f"Model: {self.model}")
            print(f"Payload: {json.dumps(data, indent=2)}")
            print("---------------------")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return content
        except Exception as e:
            print(f"Error communicating with OpenAI API: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            return f"Error: {str(e)}"

# ==== Vignette Simulator ====

class PatientSimulator:
    """Simulates a patient responding to doctor's questions based on a case database"""
    def __init__(self, case_data: Dict, model_provider: LLMProvider, patient_prompt_file: str = "prompts/patient_prompt.txt", verbose: bool = False):
        self.case_data = case_data
        self.model_provider = model_provider # Now expects a pre-configured provider
        self.history_questions = case_data.get("history_questions", [])
        self.examination_questions = case_data.get("examination_questions", [])
        self.provider_treatment_questions = case_data.get("provider_treatment_questions", [])
        self.all_questions = self.history_questions + self.examination_questions + self.provider_treatment_questions
        self.questions_by_id = {q.get("id"): q for q in self.all_questions if q.get("id")}
        
        # Load and set system prompt for the patient simulator
        try:
            with open(patient_prompt_file, 'r') as f:
                system_prompt = f.read()
            self.model_provider.set_system_prompt(system_prompt)
            self.system_prompt = system_prompt # Store the system prompt
        except Exception as e:
            print(f"Error loading patient prompt from {patient_prompt_file}: {e}")
            # Fallback to a default prompt if loading fails
            system_prompt = """"""
            sys.exit(1)
            # self.model_provider.set_system_prompt(system_prompt)
            # self.system_prompt = system_prompt
        
        self.verbose = verbose # Verbose setting is now handled by the passed provider
        # self.model_provider.set_verbose(self.verbose) # No longer needed here
        
        # --- DEBUG PRINT ---
        # print("\\n--- DEBUG: Patient Simulator Init ---")
        # print(f"Number of all_questions: {len(self.all_questions)}")
        # Print first few question IDs to verify parsing
        # if self.all_questions:
        #     print("First 5 question IDs:", [q.get('id') for q in self.all_questions[:5]])
        # print(f"Number of questions_by_id: {len(self.questions_by_id)}")

        # If there are 0 questions, print the case data and stop the program
        if len(self.all_questions) == 0:
            # print("\\n--- DEBUG: Case Data ---")
            # print(yaml.dump(self.case_data, default_flow_style=False, allow_unicode=True, width=120))
            # print("------------------------------------\n")
            sys.exit(1)

        # print("------------------------------------\n")
        # --- END DEBUG ---
    
    def _identify_relevant_question_ids(self, doctor_question: str, conversation_history: List[Dict[str, str]]) -> List[str]:
        """Step 1: Use LLM to identify relevant question IDs from the database."""
        # Prepare the list of available questions for the prompt
        available_questions_text = "\n".join(
            [f"- ID: {q.get('id')}, Question: {q.get('question_en', '')}" 
             for q in self.all_questions if q.get('id') and q.get('question_en')]
        )
        
        # Create the prompt for identifying relevant questions
        prompt = f"""
        Given the conversation history and the latest doctor question, identify the most relevant question IDs from the following list that can help answer the doctor's question. 
        Return only a JSON list of strings (the IDs), e.g., ["ID1", "ID2"]. If none are relevant or you are unsure, return [].

        Available Questions:
        {available_questions_text}

        Conversation History:
        {json.dumps(conversation_history, indent=2)}

        Doctor's Question:
        {doctor_question}
        
        Relevant Question IDs (YAML list):
        """
        
        messages = [{'role': 'user', 'content': prompt}]
        
        # Use a separate, basic system prompt for this specific task
        # Store original prompt
        original_system_prompt = self.model_provider.system_prompt
        self.model_provider.set_system_prompt("You are an AI assistant helping to find relevant information.")
        
        response = self.model_provider.chat(messages)
        
        # Restore original system prompt
        self.model_provider.set_system_prompt(original_system_prompt)
        
        # Parse the response to get the list of IDs
        try:
            # Find the JSON list in the response
            match = re.search(r'\[.*\]', response)
            if match:
                ids_str = match.group(0)
                question_ids = json.loads(ids_str)
                if isinstance(question_ids, list) and all(isinstance(id_str, str) for id_str in question_ids):
                    # print(f"Identified Question IDs: {question_ids}") # Debug print
                    return question_ids
            print(f"Warning: Could not parse question IDs from LLM response: {response}")
            return []
        except json.JSONDecodeError:
            print(f"Warning: LLM response for IDs was not valid JSON: {response}")
            return []
        except Exception as e:
            print(f"Error parsing question IDs: {e}")
            return []
            
    def _generate_response_from_context(self, doctor_question: str, relevant_ids: List[str], conversation_history: List[Dict[str, str]]) -> str:
        """Step 2: Use LLM to generate response based on history and retrieved Q/A data."""
        
        # Retrieve relevant Q/A pairs
        context_items = []
        for q_id in relevant_ids:
            q_data = self.questions_by_id.get(q_id)
            if q_data:
                context_items.append(f"  - Question (ID {q_id}): {q_data.get('question_en', '')}\n    Answer: {q_data.get('answer_en', '')}")
        
        context_string = "\n".join(context_items) if context_items else "No specific information found in your case file for this question."
        
        # Prepare messages for the LLM
        # The system prompt is already set in __init__
        user_prompt = f"""
        Based on our conversation history and the following potentially relevant information retrieved from your case file, please answer my latest question.
        Respond concisely as the patient, focusing only on answering the question asked. Use the provided case information closely.

        Retrieved Case Information:
        {context_string}

        My Question:
        {doctor_question}
        """
        
        # --- DEBUG PRINT ---
        # print("\\n--- DEBUG: Patient Response Context ---")
        # print(f"Prompt: {user_prompt}")
        # print(f"Relevant IDs used: {relevant_ids}")
        # print(f"Context String Provided:\\n{context_string}")
        # print("--------------------------------------\\n")
        # --- END DEBUG ---
        
        messages = conversation_history + [{'role': 'user', 'content': user_prompt}]
        
        # Generate the response using the main patient prompt
        response = self.model_provider.chat(messages)
        return response

    def respond_to_question(self, question: str, conversation_history: List[Dict[str, str]]) -> Tuple[str, Optional[List[str]]]:
        """
        Respond to the doctor's question using the RAG approach.
        Returns: (patient_response, list_of_question_ids_identified or None)
        """
        # Step 1: Identify relevant question IDs
        identified_ids = self._identify_relevant_question_ids(question, conversation_history)
        
        # Step 2: Generate response based on identified context (or lack thereof)
        patient_response = self._generate_response_from_context(question, identified_ids, conversation_history)
            
        # Return the response and the list of IDs identified in step 1
        return patient_response, identified_ids if identified_ids else None

class DoctorSimulator:
    """Simulates a doctor conducting a patient interview"""
    def __init__(self, model_provider: LLMProvider, doc_prompt_file: str, diagnosis_active: bool = False, verbose: bool = False):
        self.model_provider = model_provider # Now expects a pre-configured provider
        self.doc_prompt_file = doc_prompt_file # Store the prompt file path for reference/logging if needed
        self.diagnosis_active = diagnosis_active # Store the diagnosis_active flag
        
        # The system prompt is now expected to be pre-set on the model_provider 
        # by the ConversationSimulator, especially if augmented (e.g., with referral text).
        # Therefore, DoctorSimulator should not reload and set it from doc_prompt_file here.
        
        # Old code that was overwriting the prompt:
        # try:
        #     with open(doc_prompt_file, 'r') as f:
        #         doc_prompt = f.read()
        #     self.model_provider.set_system_prompt(doc_prompt)
        # except Exception as e:
        #     print(f"Error loading doctor prompt: {e}")
        #     self.model_provider.set_system_prompt("")
            
        self.verbose = verbose # Verbose setting is now handled by the passed provider
        # self.model_provider.set_verbose(self.verbose) # No longer needed here
    
    def respond_to_patient(self, conversation_history: List[Dict[str, str]]) -> str:
        """Generate the doctor's next question or response based on the conversation history"""
        if self.verbose:
            print("\n--- Doctor's System Prompt (Active for this turn) ---")
            # The system_prompt is on model_provider, which is shared and set by ConversationSimulator
            print(self.model_provider.system_prompt)
            print("-----------------------------------------------------\n")
        return self.model_provider.chat(conversation_history)
    
    def check_if_conversation_complete(self, conversation_history: List[Dict[str, str]]) -> bool:
        """Check if the doctor has completed the interview based on the mode."""
        doctor_messages = [msg for msg in conversation_history if msg["role"] == "assistant"]
        num_doctor_messages = len(doctor_messages)
        last_doctor_message = doctor_messages[-1]["content"] if doctor_messages else ""

        if not self.diagnosis_active: # Corresponds to old "summary" mode
            # Summary Mode: Check for long summary or ask about readiness for summary
            if len(last_doctor_message) > 1000:
                print("Detected long message, assuming summary complete.")
                return True
            
            if num_doctor_messages >= 16: # Initial greeting + 15 questions
                print("Reached question limit for summary mode, checking readiness...")
                messages = conversation_history + [
                    {"role": "user", "content": "Do you have enough information to provide a diagnosis summary now? Please respond only with 'YES' or 'NO'."}
                ]
                response = self.model_provider.chat(messages)
                is_ready = response.strip().upper().startswith("YES")
                print(f"LLM readiness for summary: {is_ready}")
                return is_ready
        
        elif self.diagnosis_active: # Corresponds to old "diagnosis" or "diagnosis_examination" modes
            # Diagnosis Mode: Check if the termination keyword is present anywhere in the last doctor message
            # print(f"[DEBUG] Checking for END_INTERVIEW. Last doctor message: '{last_doctor_message}'") # Debug print
            
            # Use lower() and 'in' for a case-insensitive substring check
            processed_message = last_doctor_message.lower().strip()
            target_signal = "end_interview"
            is_present = target_signal in processed_message
            
            # print(f"[DEBUG] Does message contain '{target_signal}'? {is_present}") # Explicitly print check result
            
            if is_present:
                print(f"Detected '{target_signal}' signal (case-insensitive, substring check).")
                return True

        return False

class ConversationSimulator:
    """Orchestrates the conversation between the doctor and patient simulators"""
    def __init__(self, case_file: str, prompt_dir: str, provider_name: str = "OpenAI",
                 model_name_general: str = "gpt-4o-mini", model_name_doctor_specific: Optional[str] = None,
                 verbose: bool = False, diagnosis_active: bool = False, examination_active: bool = False, 
                 treatment_active: bool = False, # New argument
                 referral_active: bool = False, # New argument for referral
                 patient_prompt_file: str = "prompts/patient_prompt.txt"):
        # Store verbose flag
        self.verbose = verbose
        self.diagnosis_active = diagnosis_active # Store the diagnosis_active flag
        self.examination_active = examination_active # Store the examination_active flag
        self.treatment_active = treatment_active # Store the treatment_active flag
        self.referral_active = referral_active # Store the referral_active flag
        self.case_file_path = case_file # Store the case file path
        self.prompt_dir = prompt_dir # Store prompt_dir
        
        # Store provider and model names
        self.provider_name = provider_name
        self.model_name_general = model_name_general
        # Use general model name for doctor if specific one is not provided
        self.model_name_doctor_specific = model_name_doctor_specific if model_name_doctor_specific else model_name_general

        # Load API keys
        api_keys = load_api_keys()
        api_key = api_keys.get(provider_name, "")
        if not api_key:
            # This check will be hit if load_api_keys didn't find a key for the provider.
            # The main function already checks and exits if the primary provider's key is missing.
            # This is more of a safeguard if ConversationSimulator is instantiated directly elsewhere.
            print(f"Error: Missing API key for provider {provider_name}. Cannot proceed.")
            raise ValueError(f"Missing API key for {provider_name}")

        # Determine the model for the doctor
        actual_model_name_doctor = self.model_name_doctor_specific if self.model_name_doctor_specific else self.model_name_general
        
        # Initialize providers with specific models
        # Provider for DoctorSimulator
        self.doctor_provider = _create_llm_provider(provider_name, actual_model_name_doctor, api_key, verbose)
        print(f"DoctorSimulator will use model: {self.doctor_provider.model} via {provider_name}")

        # Provider for PatientSimulator (and its internal RAG)
        self.patient_provider = _create_llm_provider(provider_name, self.model_name_general, api_key, verbose)
        print(f"PatientSimulator will use model: {self.patient_provider.model} via {provider_name}")

        # Provider for utility tasks (extraction, classification)
        self.utility_provider = _create_llm_provider(provider_name, self.model_name_general, api_key, verbose)
        print(f"Utility tasks (extraction, classification) will use model: {self.utility_provider.model} via {provider_name}")

        # Load case data
        try:
            with open(case_file, 'r') as f:
                self.case_data = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading case file: {e}")
            self.case_data = {}
        
        # --- DEBUG PRINT ---
        # print("\\n--- DEBUG: Case Data Loaded ---")
        # print(yaml.dump(self.case_data, default_flow_style=False, allow_unicode=True, width=120))
        # print("-----------------------------\\n")
        # --- END DEBUG ---
        
        # Determine the correct doctor prompt file based on the mode and prompt_dir
        if self.examination_active:
            prompt_file_name = "doc_prompt_diagnosis_examinations.txt"
        elif self.diagnosis_active:
            prompt_file_name = "doc_prompt_diagnosis.txt"
        else: # Default/Summary mode
            prompt_file_name = "doc_prompt.txt"
        # Add more mode-to-filename mappings here if needed
        
        effective_doc_prompt_file = str(Path(prompt_dir) / prompt_file_name)
        print(f"Using doctor prompt based on flags (diag:{self.diagnosis_active}, exam:{self.examination_active}): {effective_doc_prompt_file}")

        # Load base doctor prompt content
        base_doc_prompt_content = ""
        try:
            with open(effective_doc_prompt_file, 'r') as f:
                base_doc_prompt_content = f.read()
        except Exception as e:
            print(f"Error loading doctor prompt: {e}")
            # self.doctor_provider.set_system_prompt("") # Provider not yet fully set up, default is empty.

        # Augment doctor prompt with referral text if active
        self.active_referral_text_for_logging = None # For metadata logging
        if self.referral_active:
            referral_text_from_case = self.case_data.get('referral_text')
            if referral_text_from_case and isinstance(referral_text_from_case, str) and referral_text_from_case.strip():
                self.active_referral_text_for_logging = referral_text_from_case.strip()
                formatted_referral_info = (
                    f"\\n\\n--- START REFERRAL NOTE ---\\n"
                    f"You have received the following referral note from the patient's General Practitioner:\\n"
                    f"{self.active_referral_text_for_logging}\\n"
                    f"--- END REFERRAL NOTE ---"
                )
                base_doc_prompt_content += formatted_referral_info
                print(f"Referral mode active. Doctor's system prompt augmented with referral text from case file.")
            else:
                print(f"Referral mode active, but no 'referral_text' found or it is empty in case file: {case_file}")
        
        # Set the potentially augmented prompt on the doctor's provider
        self.doctor_provider.set_system_prompt(base_doc_prompt_content)
            
        # Initialize simulators, passing the mode and the selected prompt file, and their dedicated providers
        # DoctorSimulator's doc_prompt_file argument is now mainly for reference/logging if needed,
        # as the system prompt is already set on its provider.
        self.doctor = DoctorSimulator(self.doctor_provider, effective_doc_prompt_file, diagnosis_active=self.diagnosis_active, verbose=self.verbose)
        
        # Patient simulator uses patient_provider
        self.patient = PatientSimulator(self.case_data, self.patient_provider, patient_prompt_file=patient_prompt_file, verbose=self.verbose)
        
        # Initialize conversation storage
        self.conversation = []
        self.metadata = []
    
    def run_simulation(self) -> Dict[str, Any]:
        """Run the full conversation simulation"""
        print("Starting conversation simulation...")
        simulation_start_time_utc = datetime.now(timezone.utc).isoformat()

        # Initialize conversation and metadata
        self.conversation = []
        self.metadata = []
        current_step = 1 # Start step numbering at 1

        # Handle Referral Text (for console and metadata logging)
        # The actual referral text is part of the doctor's system prompt if active.
        # This block is for explicit logging in the output YAML and console.
        if self.referral_active and self.active_referral_text_for_logging:
            print("\\n--- Referral Note from GP (Provided to Doctor) ---")
            print(self.active_referral_text_for_logging.replace('\\n', '\\n')) # Pretty print
            print("----------------------------------------------------\\n")
            self.metadata.append({
                "step": current_step, # Or step 0 if preferred for pre-conversation info
                "speaker": "System (Referral Note)",
                "question_id": None,
                "message": self.active_referral_text_for_logging
            })
            current_step += 1


        # Patient starts the conversation using the introduction
        patient_intro = self.case_data.get('introduction', {}).get('en')
        if not patient_intro:
            # Use Hindi intro if English is missing
            patient_intro = self.case_data.get('introduction', {}).get('hi')
        if not patient_intro:
            # Fallback to a generic intro if both are missing
            patient_intro = "Hello Doctor, I need some help."
            print(f"Warning: Using default patient introduction: '{patient_intro}'")

        self.conversation = [{
            "role": "user", # Patient is the user
            "content": patient_intro
        }]
        self.metadata.append({
            "step": current_step, # Use current_step
            "speaker": "Patient",
            "question_id": None, # No question ID for the introduction
            "message": patient_intro
        })

        # Print the patient's opening message
        print(f"\n--- Step {current_step} ---")
        print("Patient: " + patient_intro.replace('\\n', '\\n         '))

        step = current_step + 1 # Next step number
        conversation_complete = False
        doctor_turn_count = 0 # DEBUG: Initialize doctor turn counter

        # Continue the conversation until it's complete
        while not conversation_complete:
            doctor_turn_count += 1 # DEBUG: Increment doctor turn counter

            # DEBUG: Print information for the doctor's 5th turn
            if doctor_turn_count == 5:
                print("\\n\\n--- DEBUG: DOCTOR LLM INPUT (Before Doctor's 5th Response) ---")
                print("Doctor's System Prompt (self.doctor_provider.system_prompt):")
                print("-------------------------------------------------------------")
                print(self.doctor_provider.system_prompt)
                print("-------------------------------------------------------------")
                print("\\nConversation History (self.conversation passed to doctor.respond_to_patient):")
                print("-------------------------------------------------------------")
                for i, msg in enumerate(self.conversation):
                    content_preview = str(msg.get('content', ''))[:200] # Increased preview length
                    print(f"  Message {i+1}: Role: {msg.get('role', 'N/A')}, Content: '{content_preview}...'")
                print("-------------------------------------------------------------")
                print("--- END DEBUG: DOCTOR LLM INPUT ---\\n\\n")


            # Doctor responds to the patient's initial statement or last message
            # Ensure the correct system prompt is set for the doctor before responding
            doc_prompt = ""
            try:
                with open(self.doctor.doc_prompt_file, 'r') as f:
                    doc_prompt = f.read()
            except Exception as e:
                print(f"Warning: Could not reload doctor prompt {self.doctor.doc_prompt_file}: {e}")
            
            # self.provider.set_system_prompt(doc_prompt) # Make sure doctor uses its prompt
            # Doctor's provider already has its system prompt set during its initialization
            # and re-set if doc_prompt_file changes (which it doesn't mid-simulation here).
            # The self.doctor.respond_to_patient call will use self.doctor_provider.
            
            doctor_response = self.doctor.respond_to_patient(self.conversation)

            self.conversation.append({
                "role": "assistant",  # Doctor is the assistant
                "content": doctor_response
            })
            self.metadata.append({
                "step": step,
                "speaker": "Doctor",
                "question_id": None,
                "message": doctor_response
            })

            # Print doctor's response
            print(f"\n--- Step {step} ---")
            print("Doctor: " + doctor_response.replace('\n', '\n        '))

            step += 1

            # Check if conversation should end (check after doctor's turn)
            if self.doctor.check_if_conversation_complete(self.conversation):
                conversation_complete = True
                break # Exit loop if complete
            
            # Ensure the correct system prompt is set for the patient before responding
            # self.patient.model_provider.set_system_prompt(self.patient.system_prompt) # Make sure patient uses its prompt
            # Patient's provider (self.patient_provider) already has its system prompt set during its initialization.

            # Patient responds to the doctor
            patient_response, question_ids = self.patient.respond_to_question(
                self.conversation[-1]["content"], # Last message was doctor's
                self.conversation
            )

            self.conversation.append({
                "role": "user",  # Patient is the user
                "content": patient_response
            })
            self.metadata.append({
                "step": step,
                "speaker": "Patient",
                "question_id": question_ids, # Store the list of IDs (can be None)
                "message": patient_response
            })

            # Print patient's response with question IDs if available
            print(f"\n--- Step {step} ---")
            print("Patient: " + patient_response.replace('\n', '\n         '))
            if question_ids: # Check if the list is not None and not empty
                print(f"Question IDs: {", ".join(question_ids)}")

            step += 1

            # Print progress
            # print(f"\nCompleted step {step-1}")

            # Add a short delay to avoid rate limiting
            time.sleep(1)

        print("\nConversation loop finished.")

        # --- Final step based on mode ---
        final_diagnosis = None
        extracted_diagnosis_json = None # Initialize extracted diagnosis variable
        
        if self.diagnosis_active: # Replaces check for mode == "diagnosis" or "diagnosis_examination"
            print("Asking final diagnosis question...")
            final_question = "[QUESTION FROM OUTSIDE EXPERT, not PATIENT] Given the conditions in this area and the kinds of patients you normally see, in your view, what would you say the patient is suffering from?"
            
            # Ensure the doctor's prompt is set
            doc_prompt = ""
            try:
                with open(self.doctor.doc_prompt_file, 'r') as f:
                    doc_prompt = f.read()
            except Exception as e:
                print(f"Warning: Could not reload doctor prompt {self.doctor.doc_prompt_file} for final question: {e}")
                # If reload fails, it uses the one set at initialization.

            # The conversation history already includes the patient's last answer.
            conversation_up_to_end_signal = self.conversation[:-1] # Exclude the last assistant message which is 'END_INTERVIEW'
            
            final_messages = conversation_up_to_end_signal + [{
                "role": "user", 
                "content": final_question
            }]
            
            # Final diagnosis question uses the doctor's provider
            # Ensure doctor's system prompt is active on its provider
            try:
                with open(self.doctor.doc_prompt_file, 'r') as f:
                    doc_prompt = f.read()
                self.doctor_provider.set_system_prompt(doc_prompt)
            except Exception as e:
                print(f"Warning: Could not reload doctor prompt {self.doctor.doc_prompt_file} for final question: {e}")
                # If reload fails, it uses the one set at initialization.

            final_diagnosis = self.doctor_provider.chat(final_messages) # Using doctor_provider
            print(f"Final Diagnosis Response:\n{final_diagnosis}")
            
            # Append final diagnosis interaction to metadata for context
            self.metadata.append({
                "step": step, 
                "speaker": "System (Final Question)", 
                "question_id": None, 
                "message": final_question
            })
            step += 1
            self.metadata.append({
                "step": step, 
                "speaker": "Doctor (Final Diagnosis)", 
                "question_id": None, 
                "message": final_diagnosis
            })

            # --- Extract the single diagnosis ---
            if final_diagnosis:
                print("\nExtracting single diagnosis...")
                try:
                    # Store doctor's prompt
                    original_system_prompt_utility = self.utility_provider.system_prompt # Using utility_provider
                    
                    # Load the extraction prompt
                    extract_prompt_file = Path(self.doctor.doc_prompt_file).parent / "extract_diagnosis_prompt.txt"
                    with open(extract_prompt_file, 'r') as f:
                        extract_prompt = f.read()
                    
                    # Set the extraction prompt on utility_provider
                    self.utility_provider.set_system_prompt(extract_prompt)
                    
                    # Prepare message for extraction
                    extract_messages = [{"role": "user", "content": final_diagnosis}]
                    
                    # Call LLM for extraction using utility_provider
                    extracted_diagnosis_json = self.utility_provider.chat(extract_messages)
                    print(f"Extracted Diagnosis JSON: {extracted_diagnosis_json}")

                    # Restore doctor's prompt (actually, utility_provider's original prompt)
                    if 'original_system_prompt_utility' in locals():
                        self.utility_provider.set_system_prompt(original_system_prompt_utility)
                    
                except Exception as e:
                    print(f"Error during diagnosis extraction: {e}")
                    extracted_diagnosis_json = f'"Unclear: Error during extraction - {e}"' # Store error in JSON format
                    # Attempt to restore prompt even on error
                    if 'original_system_prompt_utility' in locals():
                        self.utility_provider.set_system_prompt(original_system_prompt_utility)

        # Organize the results
        simulation_results = {
            "provider": self.provider_name,
            "model_general": self.model_name_general,
            "model_doctor": self.model_name_doctor_specific,
            "simulation_start_time_utc": simulation_start_time_utc,
            "diagnosis_active": self.diagnosis_active,
            "examination_active": self.examination_active,
            "treatment_active": self.treatment_active, # Add treatment_active flag
            "referral_active": self.referral_active, # Add referral_active flag
            "case_file": self.case_file_path, # Use stored case file path
            "conversation": self.metadata,
        }
        if final_diagnosis:
             simulation_results["final_diagnosis_statement"] = final_diagnosis
        if extracted_diagnosis_json: # Add extracted diagnosis if available
             simulation_results["extracted_diagnosis"] = extracted_diagnosis_json
             
        # ==== MOVED: CLASSIFY DOCTOR'S DIAGNOSIS ====
        diagnosis_classification_output = None 
        if self.diagnosis_active and "extracted_diagnosis" in simulation_results:
            # Use the value from simulation_results to ensure we're using what was stored
            current_extracted_diagnosis_json = simulation_results["extracted_diagnosis"]
            try:
                # 1. Determine and load the answer key file
                case_file_path_obj = Path(self.case_file_path)
                answer_file_name = case_file_path_obj.stem + "_answer.yaml"
                # Construct path relative to the script or a known base path if necessary
                # Assuming 'case_answers' is at the same level as 'case_definitions' or workspace root
                answer_file_path = Path("case_answers") / answer_file_name

                if not answer_file_path.is_file():
                    raise FileNotFoundError(f"Answer key file not found: {answer_file_path}")

                with open(answer_file_path, 'r') as f_ans:
                    answer_data = yaml.safe_load(f_ans)

                correct_diagnoses_list = answer_data.get("diagnosis", {}).get("correct", [])
                incorrect_diagnoses_list = answer_data.get("diagnosis", {}).get("incorrect", [])

                if not correct_diagnoses_list and not incorrect_diagnoses_list and not answer_data.get("diagnosis"):
                    raise ValueError(f"No 'diagnosis' section with 'correct' or 'incorrect' lists found in {answer_file_path}")
                # It's okay if one list is empty, but not if the whole 'diagnosis' section is malformed or missing.

                # 2. Prepare the doctor's stated diagnosis string
                doctors_stated_diagnosis = ""
                if isinstance(current_extracted_diagnosis_json, str):
                    try:
                        # Handles cases like '"flu"' (JSON string literal)
                        parsed_diag = json.loads(current_extracted_diagnosis_json)
                        if isinstance(parsed_diag, str):
                            doctors_stated_diagnosis = parsed_diag
                        # Handles cases like '{"diagnosis": "flu"}'
                        elif isinstance(parsed_diag, dict) and len(parsed_diag) == 1:
                            doctors_stated_diagnosis = str(list(parsed_diag.values())[0])
                        else: # Other JSON structures
                            doctors_stated_diagnosis = str(parsed_diag) # Fallback: stringify the structure
                    except json.JSONDecodeError:
                        # Not a valid JSON string, e.g., 'flu' or 'Error: ...'
                        # Strip potential surrounding quotes if it was meant to be a simple string.
                        doctors_stated_diagnosis = current_extracted_diagnosis_json.strip().strip('"')
                else:
                    # If it wasn't a string to begin with (should not happen with current logic but good to be safe)
                    doctors_stated_diagnosis = str(current_extracted_diagnosis_json)


                # 3. Load the classification prompt
                classification_prompt_file = Path("prompts") / "classify_diagnosis_prompt.txt"
                if not classification_prompt_file.is_file():
                    raise FileNotFoundError(f"Classification prompt file not found: {classification_prompt_file}")
                with open(classification_prompt_file, 'r') as f_prompt:
                    classification_system_prompt = f_prompt.read()

                # 4. Prepare and make the LLM call using utility_provider
                original_utility_provider_system_prompt = self.utility_provider.system_prompt # Save current utility prompt
                self.utility_provider.set_system_prompt(classification_system_prompt)

                classification_user_message_content = f"""
Doctor's Stated Diagnosis:
{doctors_stated_diagnosis}

Correct Diagnoses (from answer key):
{json.dumps(correct_diagnoses_list)}

Incorrect Diagnoses (from answer key):
{json.dumps(incorrect_diagnoses_list)}

Please classify the doctor's diagnosis and provide your confidence and explanation as a JSON object.
"""
                classification_messages = [{"role": "user", "content": classification_user_message_content}]

                if self.verbose:
                    print("\n--- Calling LLM for Diagnosis Classification ---")
                    # print(f"System Prompt: {classification_system_prompt}") # Can be very long
                    print(f"User Message Content for Classification:\n{classification_user_message_content}")

                raw_classification_response = self.utility_provider.chat(classification_messages)

                if self.verbose:
                    print(f"Raw Classification Response: {raw_classification_response}")

                self.utility_provider.set_system_prompt(original_utility_provider_system_prompt) # Restore utility prompt

                # 5. Parse the response
                try:
                    json_match = re.search(r'\{.*\}', raw_classification_response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        parsed_json_output = json.loads(json_str)
                        
                        # Validate essential keys
                        required_keys = ["classification", "confidence", "explanation"]
                        if not all(key in parsed_json_output for key in required_keys):
                            raise ValueError(f"Classification JSON missing one or more required keys: {required_keys}")

                        # Ensure confidence is null if classification is "unclear"
                        if parsed_json_output.get("classification") == "unclear":
                            parsed_json_output["confidence"] = None
                        
                        diagnosis_classification_output = parsed_json_output
                    else:
                        raise json.JSONDecodeError("No JSON object found in LLM response", raw_classification_response, 0)

                except json.JSONDecodeError as je:
                    error_msg = f"Failed to parse classification JSON from LLM. Response: {raw_classification_response}. Error: {je}"
                    print(f"Warning: {error_msg}")
                    diagnosis_classification_output = {"classification": "Error", "confidence": None, "explanation": error_msg}
                except ValueError as ve: # For missing keys or other validation issues
                    error_msg = f"Invalid classification JSON structure from LLM. Response: {raw_classification_response}. Error: {ve}"
                    print(f"Warning: {error_msg}")
                    diagnosis_classification_output = {"classification": "Error", "confidence": None, "explanation": error_msg}

            except FileNotFoundError as fnf_ex:
                print(f"Error during diagnosis classification setup (file not found): {fnf_ex}")
                diagnosis_classification_output = {"classification": "Error", "confidence": None, "explanation": str(fnf_ex)}
            except ValueError as val_ex: # Catch issues like no diagnoses in answer key
                print(f"Error during diagnosis classification setup (value error): {val_ex}")
                diagnosis_classification_output = {"classification": "Error", "confidence": None, "explanation": str(val_ex)}
            except Exception as e:
                error_msg = f"Unexpected error during diagnosis classification: {e}"
                print(error_msg)
                # import traceback # Consider adding for more detailed debugging if needed
                # traceback.print_exc()
                diagnosis_classification_output = {"classification": "Error", "confidence": None, "explanation": error_msg}
        
        if diagnosis_classification_output:
            simulation_results["diagnosis_classification_details"] = diagnosis_classification_output
            
        # ==== ADDED: TREATMENT EXTRACTION AND CLASSIFICATION ====
        extracted_treatments_list = None
        treatment_classification_output = None

        if self.treatment_active:
            print("\n--- Treatment Phase ---")
            # 1. Ask doctor for treatment recommendation
            treatment_question = "What medicine(s) would you give to this patient?"
            print(f"Asking treatment question: {treatment_question}")

            # Use doctor's provider and prompt for this question
            # The conversation history up to this point is self.conversation
            # If diagnosis was active, final_diagnosis might have been added to metadata but not self.conversation for LLM.
            # We should use the main conversation history.
            
            # Ensure doctor's system prompt is active on its provider
            try:
                with open(self.doctor.doc_prompt_file, 'r') as f:
                    doc_prompt_for_treatment_q = f.read()
                self.doctor_provider.set_system_prompt(doc_prompt_for_treatment_q)
            except Exception as e:
                print(f"Warning: Could not reload doctor prompt {self.doctor.doc_prompt_file} for treatment question: {e}")
            
            treatment_messages = self.conversation + [{"role": "user", "content": treatment_question}]
            doctors_treatment_response = self.doctor_provider.chat(treatment_messages)
            print(f"Doctor's Treatment Response:\\n{doctors_treatment_response}")

            self.metadata.append({
                "step": step,
                "speaker": "System (Treatment Question)",
                "question_id": None,
                "message": treatment_question
            })
            step += 1
            self.metadata.append({
                "step": step,
                "speaker": "Doctor (Treatment Response)",
                "question_id": None,
                "message": doctors_treatment_response
            })
            step += 1


            # 2. Extract treatments from doctor's response
            if doctors_treatment_response:
                print("\nExtracting treatments...")
                try:
                    original_utility_prompt = self.utility_provider.system_prompt
                    extract_treat_prompt_file = Path(self.prompt_dir) / "extract_treatment_prompt.txt"
                    with open(extract_treat_prompt_file, 'r') as f:
                        extract_treat_prompt = f.read()
                    
                    # Substitute placeholder in prompt if any (currently {{DOCTOR_STATEMENT_CONTENT}})
                    extract_treat_prompt_filled = extract_treat_prompt.replace("{{DOCTOR_STATEMENT_CONTENT}}", doctors_treatment_response)

                    self.utility_provider.set_system_prompt(extract_treat_prompt_filled) # System prompt is the full filled prompt here
                    
                    # The user message is now part of the system prompt for this specific prompt structure
                    extract_treat_messages = [{"role": "user", "content": doctors_treatment_response}] # Content here is doctor's response

                    # The actual prompt content is now in the system prompt.
                    # For this specific extraction prompt, the doctor's statement is part of the user message to the LLM
                    # The system prompt guides the overall task.
                    # Let's adjust: system prompt is the generic instruction, user message contains the text to process.
                    self.utility_provider.set_system_prompt(extract_treat_prompt) # Set the generic system prompt
                    
                    # The prompt has {{DOCTOR_STATEMENT_CONTENT}}. This needs to be replaced.
                    final_extraction_user_message = doctors_treatment_response # The part that replaces {{DOCTOR_STATEMENT_CONTENT}}
                                                                            # No, the prompt itself contains this structure.
                                                                            # The LLM expects a completion.

                    # Let's assume the prompt file is used AS IS, and the LLM completes it.
                    # This means the entire prompt file is the "user" message, and the LLM provides the completion.
                    # This matches how some models work with specific "text completion" vs "chat" endpoints.
                    # Given we are using a "chat" endpoint, we should provide a system prompt and a user message.

                    # Let's use the content of `extract_treatment_prompt.txt` (which has the placeholder) as the user message.
                    # And a very simple system prompt.
                    
                    self.utility_provider.set_system_prompt("You are a helpful AI assistant that follows instructions precisely.") # Basic system prompt
                    
                    # Load the prompt template from the file
                    with open(extract_treat_prompt_file, 'r') as f:
                        extract_treat_system_prompt = f.read()

                    user_message_for_extraction = extract_treat_system_prompt.replace("{{DOCTOR_STATEMENT_CONTENT}}", doctors_treatment_response)
                    extract_treat_messages = [{"role": "user", "content": user_message_for_extraction}]

                    raw_extracted_treatments = self.utility_provider.chat(extract_treat_messages)
                    print(f"Raw Extracted Treatments: {raw_extracted_treatments}")

                    # Try to parse the JSON list from the response
                    match = re.search(r'\[(.*?)\]', raw_extracted_treatments, re.DOTALL)
                    if match:
                        json_list_str = match.group(0)
                        try:
                            extracted_treatments_list = json.loads(json_list_str)
                            if not isinstance(extracted_treatments_list, list):
                                print(f"Warning: Extracted treatments is not a list: {extracted_treatments_list}")
                                extracted_treatments_list = [str(extracted_treatments_list)] # convert to list
                        except json.JSONDecodeError:
                            print(f"Warning: Could not parse extracted treatments JSON: {json_list_str}")
                            extracted_treatments_list = [raw_extracted_treatments] # Store raw as list item
                    else:
                        print(f"Warning: No JSON list found in treatment extraction response: {raw_extracted_treatments}")
                        extracted_treatments_list = [raw_extracted_treatments] # Store raw as list item
                    
                    simulation_results["extracted_treatments"] = extracted_treatments_list
                    self.utility_provider.set_system_prompt(original_utility_prompt) # Restore

                except Exception as e:
                    print(f"Error during treatment extraction: {e}")
                    simulation_results["extracted_treatments"] = [f"Error during extraction: {e}"]
                    if 'original_utility_prompt' in locals() and self.utility_provider:
                         self.utility_provider.set_system_prompt(original_utility_prompt)

            # 3. Classify extracted treatments
            if extracted_treatments_list:
                print("\nClassifying extracted treatments individually...")
                individual_classification_details = []
                final_correct_count = 0
                final_palliative_count = 0
                final_unnecessary_count = 0
                
                try:
                    original_utility_prompt_classify = self.utility_provider.system_prompt
                    
                    # Load answer key for treatments (remains the same)
                    case_file_path_obj = Path(self.case_file_path)
                    answer_file_name = case_file_path_obj.stem + "_answer.yaml"
                    answer_file_path = Path("case_answers") / answer_file_name

                    if not answer_file_path.is_file():
                        raise FileNotFoundError(f"Treatment answer key file not found: {answer_file_path}")

                    with open(answer_file_path, 'r') as f_ans_treat:
                        treatment_answer_data = yaml.safe_load(f_ans_treat)
                    
                    correct_treats = treatment_answer_data.get("treatments", {}).get("correct", [])
                    palliative_treats = treatment_answer_data.get("treatments", {}).get("palliative", [])
                    unnecessary_treats = treatment_answer_data.get("treatments", {}).get("unnecessary_or_harmful", [])

                    if not any([correct_treats, palliative_treats, unnecessary_treats]) and not treatment_answer_data.get("treatments"):
                         raise ValueError(f"No 'treatments' section with 'correct', 'palliative', or 'unnecessary_or_harmful' lists found in {answer_file_path}")

                    # Load the NEW classification prompt for a single treatment
                    classify_single_treat_prompt_file = Path(self.prompt_dir) / "classify_single_treatment_prompt.txt" # New prompt file
                    if not classify_single_treat_prompt_file.is_file():
                        raise FileNotFoundError(f"Single treatment classification prompt file not found: {classify_single_treat_prompt_file}")
                    with open(classify_single_treat_prompt_file, 'r') as f_prompt_classify:
                        classify_single_treat_template = f_prompt_classify.read()
                    
                    self.utility_provider.set_system_prompt("You are a helpful AI assistant that follows instructions precisely and returns only a valid JSON object.")

                    for stated_treatment in extracted_treatments_list:
                        if not stated_treatment or str(stated_treatment).strip() == "": # Skip empty treatments
                            continue

                        print(f"  Classifying: '{stated_treatment}'")
                        
                        # Fill placeholders in the single treatment classification prompt
                        user_message_for_single_classification = (
                            classify_single_treat_template
                            .replace("{{DOCTORS_STATED_TREATMENT}}", str(stated_treatment))
                            .replace("{{CORRECT_TREATMENTS_JSON_LIST}}", json.dumps(correct_treats))
                            .replace("{{PALLIATIVE_TREATMENTS_JSON_LIST}}", json.dumps(palliative_treats))
                            .replace("{{UNNECESSARY_OR_HARMFUL_TREATMENTS_JSON_LIST}}", json.dumps(unnecessary_treats))
                        )

                        classify_treat_messages = [{"role": "user", "content": user_message_for_single_classification}]

                        if self.verbose:
                            print(f"    User Message for single treatment classification:\\n{user_message_for_single_classification}")

                        raw_single_classification_response = self.utility_provider.chat(classify_treat_messages)

                        if self.verbose:
                            print(f"    Raw single classification response: {raw_single_classification_response}")
                        
                        # Parse the JSON response for the single treatment
                        try:
                            json_match_single = re.search(r'{.*}', raw_single_classification_response, re.DOTALL)
                            if json_match_single:
                                json_str_single = json_match_single.group(0)
                                single_class_result = json.loads(json_str_single)
                                
                                # Validate required keys from the LLM for this single item
                                required_single_keys = ["stated_treatment", "category", "matched_key_item", "explanation"]
                                if not all(key in single_class_result for key in required_single_keys):
                                    raise ValueError(f"Single treatment classification JSON missing one or more required keys: {required_single_keys}. Response: {json_str_single}")

                                # Ensure category is one of the expected values
                                valid_categories = ["correct", "palliative", "unnecessary_or_harmful", "not found"]
                                if single_class_result.get("category") not in valid_categories:
                                     print(f"Warning: LLM returned invalid category '{single_class_result.get('category')}' for treatment '{stated_treatment}'. Defaulting to 'not found'.")
                                     single_class_result["category"] = "not found"
                                     single_class_result["matched_key_item"] = "not found"


                                individual_classification_details.append({
                                    "stated_treatment": single_class_result.get("stated_treatment", str(stated_treatment)), # Use original if LLM omits
                                    "category": single_class_result.get("category"),
                                    "matched_key_item": single_class_result.get("matched_key_item")
                                    # Explanation for single treatment is not directly added to the list here,
                                    # but could be logged or used if needed. The overall explanation will be generic.
                                })
                                
                                # Update counts
                                category = single_class_result.get("category")
                                if category == "correct":
                                    final_correct_count += 1
                                elif category == "palliative":
                                    final_palliative_count += 1
                                elif category == "unnecessary_or_harmful":
                                    final_unnecessary_count += 1
                                # "not found" does not increment these specific counts

                            else:
                                raise json.JSONDecodeError("No JSON object found in LLM single treatment classification response", raw_single_classification_response, 0)
                        
                        except (json.JSONDecodeError, ValueError) as e_parse_single:
                            error_msg_single = f"Failed to parse/validate classification for treatment '{stated_treatment}'. Response: {raw_single_classification_response}. Error: {e_parse_single}"
                            print(f"Warning: {error_msg_single}")
                            individual_classification_details.append({
                                "stated_treatment": str(stated_treatment),
                                "category": "error_parsing_classification",
                                "matched_key_item": "error"
                            })
                    
                    self.utility_provider.set_system_prompt(original_utility_prompt_classify) # Restore original utility prompt

                    # Assemble the final treatment_classification_output
                    treatment_classification_output = {
                        "correct_count": final_correct_count,
                        "palliative_count": final_palliative_count,
                        "unnecessary_or_harmful_count": final_unnecessary_count,
                        "classification_details": individual_classification_details,
                        "explanation": "Each stated treatment was individually classified against the answer key categories. Counts reflect these individual classifications."
                    }

                except FileNotFoundError as fnf_ex_treat:
                    print(f"Error during treatment classification setup (file not found): {fnf_ex_treat}")
                    treatment_classification_output = {"error_summary": f"File not found: {fnf_ex_treat}"}
                except ValueError as val_ex_treat: # Catches issues like missing sections in answer key
                    print(f"Error during treatment classification setup (value error): {val_ex_treat}")
                    treatment_classification_output = {"error_summary": f"Value error: {val_ex_treat}"}
                # Removed the all-encompassing JSONDecodeError and Exception catchers here,
                # as individual parsing errors are handled per-treatment.
                # The outer try-except here is more for setup errors (file loading, answer key issues).
                except Exception as e_treat_setup:
                    error_msg_treat_gen = f"Unexpected error during treatment classification setup: {e_treat_setup}"
                    print(error_msg_treat_gen)
                    treatment_classification_output = {"error_summary": error_msg_treat_gen}
                finally: # Ensure utility_provider prompt is restored if it was set
                    if 'original_utility_prompt_classify' in locals() and hasattr(self, 'utility_provider') and self.utility_provider:
                        self.utility_provider.set_system_prompt(original_utility_prompt_classify)
            
            if treatment_classification_output:
                 simulation_results["treatment_classification_details"] = treatment_classification_output
        
        simulation_end_time_utc = datetime.now(timezone.utc).isoformat()
        simulation_results["simulation_end_time_utc"] = simulation_end_time_utc
        
        # Optionally include raw messages if needed for debugging
        # simulation_results["raw_messages"] = self.conversation

        return simulation_results
    
    def save_results(self, output_file: str, results: Dict[str, Any]):
        """Save the simulation results to a YAML file"""
        try:
            # Ensure the output directory exists
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                # Use allow_unicode=True for broader character support
                # Use sort_keys=False to maintain insertion order
                yaml.dump(results, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")

def run_simulation_task(task_args: Tuple[int, argparse.Namespace, str, str, int, Optional[str]]):
    """
    Worker function to run a single simulation for a specific case and doctor model.
    Designed to be called by multiprocessing.Pool.map.
    task_args contains: (simulation_id_for_case, main_cli_args, current_case_file_path, main_batch_output_dir_path, num_sims_for_this_case, current_doctor_model)
    """
    simulation_id, main_args, current_case_file, main_batch_output_dir_str, total_sims_for_case, current_doctor_model = task_args

    # Python's default print is not thread-safe / process-safe for interleaved output.
    # For critical logging from parallel processes, consider using the logging module
    # or writing to process-specific files. For this script, interleaved stdout is accepted.
    
    case_stem = Path(current_case_file).stem
    
    # Determine effective doctor model for this specific task instance
    # This logic is also used for the `instance_log_prefix`
    _effective_doctor_model_for_task = current_doctor_model if current_doctor_model and current_doctor_model.strip() and current_doctor_model.lower() != 'none' else main_args.model

    filename_doctor_model_suffix = ""
    if _effective_doctor_model_for_task != main_args.model:
        # Sanitize model name for filename
        sanitized_model_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', _effective_doctor_model_for_task)
        filename_doctor_model_suffix = f"_drmodel_{sanitized_model_name}"

    # Construct the expected output filename
    output_filename_for_sim = Path(main_batch_output_dir_str) / f"{case_stem}_sim_{simulation_id}{filename_doctor_model_suffix}.yaml"
    
    # Construct a unique identifier for this simulation instance for logging
    doctor_model_name_for_log = _effective_doctor_model_for_task # Use the already determined effective model
    instance_log_prefix = f"Case '{case_stem}', Sim {simulation_id}/{total_sims_for_case}, DrModel: {doctor_model_name_for_log} (PID: {os.getpid()})"

    # Check if file exists and if we are in continue mode
    if main_args.continue_batch and output_filename_for_sim.exists():
        sys.stdout.write(f"--- SKIPPING (already exists in continue mode): {instance_log_prefix}. Output: {output_filename_for_sim} ---\n")
        sys.stdout.flush()
        return True # Indicate success as it's already done
    
    sys.stdout.write(f"--- Starting: {instance_log_prefix} ---\n")
    sys.stdout.flush()
    
    try:
        # Use the specific doctor model for this task, or general model if current_doctor_model is None/empty/'none'
        # This is already captured in _effective_doctor_model_for_task, use that for clarity.
        # effective_doctor_model = current_doctor_model if current_doctor_model and current_doctor_model.strip() and current_doctor_model.lower() != 'none' else main_args.model

        simulator = ConversationSimulator(
            case_file=current_case_file, # Use the specific case for this task
            prompt_dir=main_args.prompt_dir,
            provider_name=main_args.provider,
            model_name_general=main_args.model, # General model
            model_name_doctor_specific=_effective_doctor_model_for_task, # Pass the resolved doctor model
            verbose=main_args.verbose,
            diagnosis_active=main_args.diagnosis, # Pass new flag
            examination_active=main_args.examination, # Pass new flag
            treatment_active=main_args.treatment, # Pass new flag
            referral_active=main_args.referral, # Pass new flag
            patient_prompt_file=main_args.patient_prompt
        )

        results = simulator.run_simulation()
        
        # Add simulation_id, case_file, and doctor_model to the results for clarity
        results["simulation_id_for_case"] = simulation_id
        results["original_case_file"] = current_case_file
        results["doctor_model_used"] = _effective_doctor_model_for_task # Log the model actually used
        
        # Output filename determined earlier as output_filename_for_sim
        simulator.save_results(str(output_filename_for_sim), results)
        sys.stdout.write(f"--- Finished: {instance_log_prefix}. Results saved to {output_filename_for_sim} ---\n")
        sys.stdout.flush()
        return True # Indicate success
    except Exception as e:
        sys.stderr.write(f"!!! ERROR in {instance_log_prefix} !!!: {e}\n")
        import traceback
        sys.stderr.write(traceback.format_exc() + "\n")
        sys.stderr.flush()
        return False # Indicate failure

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Vignette Simulator for doctor-patient conversations")
    parser.add_argument(
        "--cases", 
        type=str, 
        nargs='+', 
        required=True, 
        help="Path(s) to the case definition file(s) to process (e.g., case_definitions/my_case.yaml)"
    )
    parser.add_argument("--prompt-dir", type=str, default="prompts", help="Directory containing prompt files")
    parser.add_argument("--provider", type=str, default="OpenAI", choices=["OpenAI", "Anthropic"], help="LLM provider to use")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use")
    parser.add_argument(
        "--model-doctors", 
        type=str, 
        nargs='*',  # Zero or more arguments
        default=None,  # Default to None if not provided
        help="Specific model(s) for the doctor. If not set, or if an element is 'None' or empty, the general model specified by --model is used for that doctor iteration."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output, printing full prompts")
    parser.add_argument("--diagnosis", action="store_true", help="Enable diagnosis-specific behavior (final question, extraction, classification) and use diagnosis prompt if --examination is not set.")
    parser.add_argument("--examination", action="store_true", help="Enable examination-specific behavior (uses examination prompt) and implies diagnosis behavior.")
    parser.add_argument("--treatment", action="store_true", help="Enable treatment recommendation, extraction, and classification phase.")
    parser.add_argument("--referral", action="store_true", help="Enable referral mode. If active, referral text from case file is added to doctor's system prompt.")
    parser.add_argument("--n_sims", type=int, default=1, help="Number of simulations to run PER CASE")
    parser.add_argument("--patient-prompt", type=str, default="prompts/patient_prompt.txt", help="Path to the patient prompt file")
    parser.add_argument(
        "--continue-batch",
        type=str,
        default=None,
        help="Path to an existing timestamped output directory to continue. If set, this directory will be used, and simulations with existing output YAML files will be skipped."
    )

    args = parser.parse_args()

    # Check API keys
    api_keys = load_api_keys()
    if not api_keys.get(args.provider):
        print(f"Error: No API key found for {args.provider}.")
        print("Please set the API key in api_keys.yaml or in the .env file.")
        print(f"For OpenAI: OPENAI_API_KEY environment variable")
        print(f"For Anthropic: ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    overall_cases_processed = 0
    overall_cases_succeeded = 0
    overall_cases_failed = 0

    # Create base output directory
    base_output_dir = Path("outputs")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped subdirectory for this entire batch of cases
    # unless --continue-batch is specified
    if args.continue_batch:
        batch_output_dir = Path(args.continue_batch)
        if not batch_output_dir.is_dir():
            print(f"Error: Provided --continue-batch directory '{args.continue_batch}' does not exist or is not a directory.")
            sys.exit(1)
        print(f"Continuing batch in existing directory: {batch_output_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_output_dir = base_output_dir / timestamp
        batch_output_dir.mkdir(parents=True, exist_ok=True) # Use parents=True, exist_ok=True
        print(f"New batch output directory: {batch_output_dir}")


    if not args.cases:
        print("No case files provided. Exiting.")
        sys.exit(0)

    # Determine the list of doctor models to iterate through
    # If --model-doctors is not provided (args.model_doctors is None), run once with None (which means use general model)
    # If --model-doctors is provided as an empty list (e.g. --model-doctors), also run once with None.
    # If --model-doctors is ['modelA', 'modelB'], run for modelA and modelB.
    # If an element in the list is "None" (string) or empty string, it will be treated as using the general model.
    doctor_models_to_run = args.model_doctors
    if doctor_models_to_run is None or not doctor_models_to_run: # Handles None or empty list
        doctor_models_to_run = [None] # Represents using the general model for the doctor

    total_iterations = len(args.cases) * args.n_sims * len(doctor_models_to_run)
    current_iteration = 0
    print(f"Total simulation iterations to perform (cases * n_sims * doctor_models): {total_iterations}")


    for case_idx, current_case_file_path in enumerate(args.cases):
        overall_cases_processed += 1 # This counts unique case files
        print(f"\n--- Processing Case {case_idx + 1} of {len(args.cases)}: {current_case_file_path} ---")
        
        if not Path(current_case_file_path).is_file():
            print(f"Error: Case file not found: {current_case_file_path}. Skipping this case.")
            overall_cases_failed += 1 # This failure is at the case level
            continue

        case_successful_for_all_doctor_models = True

        for doc_model_idx, current_doctor_model in enumerate(doctor_models_to_run):
            doctor_model_name_for_print = current_doctor_model if current_doctor_model and current_doctor_model.strip() and current_doctor_model.lower() != 'none' else args.model
            print(f"  --- Iterating with Doctor Model {doc_model_idx + 1} of {len(doctor_models_to_run)}: {doctor_model_name_for_print} ---")

            sims_for_this_case_and_model = args.n_sims
            tasks_to_run_for_current_config = []
            for i in range(1, sims_for_this_case_and_model + 1):
                current_iteration +=1
                # task_args now includes current_doctor_model
                tasks_to_run_for_current_config.append((i, args, current_case_file_path, str(batch_output_dir), sims_for_this_case_and_model, current_doctor_model))

            current_config_succeeded_all_sims = True # Assume success for this case+model config

            if sims_for_this_case_and_model == 0:
                print(f"    No simulations to run for case {current_case_file_path} with doctor model {doctor_model_name_for_print} (n_sims is 0). Marked as success for this configuration.")
                # current_config_succeeded_all_sims remains true
            elif sims_for_this_case_and_model == 1:
                print(f"    Running a single simulation (Iter {current_iteration}/{total_iterations}) for case {current_case_file_path}, doctor model {doctor_model_name_for_print} sequentially...")
                success_status = run_simulation_task(tasks_to_run_for_current_config[0])
                if not success_status:
                    current_config_succeeded_all_sims = False
                    print(f"    Single simulation failed for case {current_case_file_path}, doctor model {doctor_model_name_for_print}.")
            else: # sims_for_this_case_and_model > 1 (parallel execution)
                cpu_cores = os.cpu_count()
                num_workers = min(sims_for_this_case_and_model, cpu_cores if cpu_cores and cpu_cores > 0 else 1)
                print(f"    Starting {sims_for_this_case_and_model} simulations (Iters {current_iteration - sims_for_this_case_and_model + 1} to {current_iteration}/{total_iterations}) for case {current_case_file_path}, doctor model {doctor_model_name_for_print} using {num_workers} parallel processes...")
                
                with multiprocessing.Pool(processes=num_workers) as pool:
                    simulation_outcomes = pool.map(run_simulation_task, tasks_to_run_for_current_config)
                
                actual_successful_sims_count = sum(1 for outcome in simulation_outcomes if outcome)
                failed_sims_count = sims_for_this_case_and_model - actual_successful_sims_count
                
                print(f"    --- Summary for case: {current_case_file_path}, Doctor Model: {doctor_model_name_for_print} ---")
                print(f"    Total simulations attempted: {sims_for_this_case_and_model}")
                print(f"    Successfully completed: {actual_successful_sims_count}")
                print(f"    Failed: {failed_sims_count}")
                
                if failed_sims_count > 0:
                    current_config_succeeded_all_sims = False
                    print(f"    Please check output above for error details from failed simulations for this configuration.")
            
            if not current_config_succeeded_all_sims:
                case_successful_for_all_doctor_models = False
                # We don't increment overall_cases_failed here yet, that's done after all doc_models for a case
        
        # After iterating all doctor models for the current_case_file_path
        if case_successful_for_all_doctor_models:
            overall_cases_succeeded +=1
            print(f"--- Case {current_case_file_path} processed successfully for all doctor models ({sims_for_this_case_and_model} sims each) ---")
        else:
            overall_cases_failed += 1
            print(f"--- Case {current_case_file_path} had one or more failures across different doctor models ---")

    # Main orchestration summary
    print(f"\n\n--- Overall Batch Summary ---")
    print(f"Total cases specified: {len(args.cases)}")
    print(f"Total cases attempted processing: {overall_cases_processed}")
    print(f"Cases fully successful: {overall_cases_succeeded}")
    print(f"Cases with failures: {overall_cases_failed}")
    
    # This print is CRITICAL for run.sh
    print(f"Results saved in: {batch_output_dir}") 

    if overall_cases_failed > 0:
        print(f"\nExiting with error code due to {overall_cases_failed} case failure(s).")
        sys.exit(1)
    else:
        print("\nAll specified cases processed successfully.")
        sys.exit(0)

if __name__ == "__main__":
    # This check is important for multiprocessing on Windows, and good practice otherwise
    main()