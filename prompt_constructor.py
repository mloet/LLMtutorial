import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import csv
from typing import Dict, List, Optional
import argparse

class SyntheaPromptGenerator:
    def __init__(self, data_dir: str):
        """
        Initialize Synthea data prompt generator
        
        Args:
            data_dir (str): Directory containing Synthea CSV files
        """
        self.data_dir = Path(data_dir)
        self.encounters = None
        self.conditions = None
        self.medications = None
        self.patients = None
        self.careplans = None
        self.observations = None
        
    def load_data(self):
        try:
            self.encounters = pd.read_csv(self.data_dir / 'encounters.csv')
            self.conditions = pd.read_csv(self.data_dir / 'conditions.csv')
            self.medications = pd.read_csv(self.data_dir / 'medications.csv')
            self.patients = pd.read_csv(self.data_dir / 'patients.csv')
            self.careplans = pd.read_csv(self.data_dir / 'careplans.csv')
            self.observations = pd.read_csv(self.data_dir / 'observations.csv')
            
            date_columns = {
                'encounters': ['START', 'STOP'],
                'conditions': ['START', 'STOP'],
                'medications': ['START', 'STOP'],
                'careplans': ['START', 'STOP'],
                'patients': ['BIRTHDATE', 'DEATHDATE'],
                'observations': ['DATE']
            }
            
            for df_name, cols in date_columns.items():
                df = getattr(self, df_name)
                for col in cols:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col]).dt.tz_localize(None)
                        
        except FileNotFoundError as e:
            print(f"Error loading Synthea data: {e}")
            raise

    def get_patient_history(self, patient_id: str, encounter_id: str) -> Dict:
        """
        Get comprehensive patient history for prompts
        """
        patient = self.patients[self.patients['Id'] == patient_id].iloc[0]
        patient_conditions = self.conditions[self.conditions['PATIENT'] == patient_id]
        patient_medications = self.medications[self.medications['PATIENT'] == patient_id]
        patient_encounters = self.encounters[self.encounters['PATIENT'] == patient_id]
        patient_careplans = self.careplans[self.careplans['PATIENT'] == patient_id]
        encounter = self.encounters[self.encounters['Id'] == encounter_id].iloc[0]
        observations = self.observations[(self.observations['PATIENT'] == patient_id) &
                                              (self.observations['ENCOUNTER'] == encounter_id)]
        

        
        history = {
            'demographics': {
                'age': (encounter['START'].year - patient['BIRTHDATE'].year),
                'gender': patient['GENDER'],
                'race': patient['RACE'],
                'ethnicity': patient['ETHNICITY']
            },
            'observations': self._process_observations(observations),
            'conditions': patient_conditions['DESCRIPTION'].tolist(),
            'medications': patient_medications['DESCRIPTION'].tolist(),
            'careplans': patient_careplans['DESCRIPTION'].tolist(),
            'encounters': patient_encounters['DESCRIPTION'].tolist()
        }
        
        return history

    def generate_chain_of_thought_prompt(self, encounter_id: str) -> str:
        encounter = self.encounters[self.encounters['Id'] == encounter_id].iloc[0]
        history = self.get_patient_history(encounter['PATIENT'], encounter_id)
    
        prompt = f"""Analyze this patient's medical history step by step:
Patient Information:
- Age: {history['demographics']['age']}
- Gender: {history['demographics']['gender']}
- Race: {history['demographics']['race']}
- Ethnicity: {history['demographics']['ethnicity']}

Medical History:
{self._format_list('Conditions\n', history['conditions'])}
{self._format_list('Current Medications\n', history['medications'])}
{self._format_list('Observations\n', history['observations'])}

Reason through this case:
1. What are the primary health concerns based on the conditions?
2. How do these conditions interact with each other?
3. Are the current medications appropriate for these conditions?
4. What additional monitoring or interventions might be needed?
5. What preventive measures should be considered?

Based on this analysis, please provide:
1. A prioritized list of health concerns
2. Recommendations for medication adjustments
3. Suggested monitoring plan
4. Preventive care recommendations
"""
        return prompt

    def generate_tree_of_thoughts_prompt(self, encounter_id: str) -> str:
        """
        Generate a tree-of-thoughts prompt for complex medical decision making
        """
        encounter = self.encounters[self.encounters['Id'] == encounter_id].iloc[0]
        history = self.get_patient_history(encounter['PATIENT'], encounter_id)
        
        prompt = f"""Let's analyze this patient's case through multiple reasoning paths:
Patient Profile:
{self._format_dict(history['demographics'])}

Current Medical Status:
{self._format_list('Active Conditions\n', history['conditions'])}
{self._format_list('Current Medications\n', history['medications'])}
{self._format_list('Observations\n', history['observations'])}

Path A - Diagnostic Assessment:
1. What patterns emerge from the current conditions?
2. Are there potential underlying conditions?
3. What are the risk factors for disease progression?

Path B - Treatment Optimization:
1. How effective is the current medication regimen?
2. What are potential drug interactions?
3. What alternative treatments could be considered?

Path C - Preventive Planning:
1. What are the modifiable risk factors?
2. What preventive screenings are needed?
3. What lifestyle modifications would be beneficial?

Synthesize these paths to provide:
1. Comprehensive health assessment
2. Treatment recommendations
3. Preventive care plan
"""
        return prompt

    def generate_few_shot_prompt(self, encounter_id: str) -> str:
        encounter = self.encounters[self.encounters['Id'] == encounter_id].iloc[0]
        target_history = self.get_patient_history(encounter['PATIENT'], encounter_id)
        
        prompt = """Here is a similar medical case and its analysis:\n
Patient Profile:
- Age: 45
- Gender: F
- Race: Black
- Ethnicity: Non-Hispanic
Conditions:
- Housing instability (finding)
- Financial insecurity (finding)
- Social isolation (finding)
- Stress (finding)
- Hypertension (disorder)
- Type 2 diabetes mellitus (disorder)
- Depression (disorder)
- Obesity (finding)
Current Medications:
- Metformin 500 mg oral tablet
- Lisinopril 10 mg oral tablet
- Sertraline 50 mg oral tablet

Analysis:
The patient presents with multiple socio-environmental challenges, including housing instability, 
financial insecurity, and social isolation, contributing to significant stress. These factors are 
likely exacerbating her chronic conditions of hypertension, type 2 diabetes, and depression. The 
presence of obesity adds further risk for complications.\n\n"""
        
        prompt += f"""Now analyze this new case:
Patient Profile:
{self._format_dict(target_history['demographics'])}
{self._format_list('Conditions\n', target_history['conditions'])}
{self._format_list('Current Medications\n', target_history['medications'])}
{self._format_list('Observations\n', target_history['observations'])}

Provide a similar analysis for this case.
"""
        return prompt

    def generate_medical_note_prompt(self, encounter_id: str, language: str) -> str:
        """
        Generate prompt for medical note simplification
        """
        encounter = self.encounters[self.encounters['Id'] == encounter_id].iloc[0]
        patient_id = encounter['PATIENT']
        patient = self.patients[self.patients['Id'] == patient_id].iloc[0]
        
        encounter_conditions = self.conditions[
            (self.conditions['PATIENT'] == patient_id) &
            (self.conditions['START'] <= encounter['START']) &
            ((self.conditions['STOP'].isna()) | (self.conditions['STOP'] >= encounter['START']))
        ]
        
        encounter_medications = self.medications[
            (self.medications['PATIENT'] == patient_id) &
            (self.medications['START'] <= encounter['START']) &
            ((self.medications['STOP'].isna()) | (self.medications['STOP'] >= encounter['START']))
        ]

        ongoing_careplans = self.careplans[
            (self.careplans['PATIENT'] == patient_id) &
            (self.careplans['START'] <= encounter['START']) &
            ((self.careplans['STOP'].isna()) | (self.careplans['STOP'] >= encounter['START']))
        ]

        encounter_careplans = self.careplans[
            (self.careplans['PATIENT'] == patient_id) &
            (self.careplans['ENCOUNTER'] == encounter_id)
        ]
        
        note = f"""MEDICAL NOTE
Date: {encounter['START'].strftime('%Y-%m-%d')}
Visit Type: {encounter['DESCRIPTION']}

PATIENT DEMOGRAPHICS
Age: {(encounter['START'].year - patient['BIRTHDATE'].year)}
Gender: {patient['GENDER']}

REASON FOR VISIT
{'Not provided' if pd.isna(encounter['REASONDESCRIPTION']) else encounter['REASONDESCRIPTION']}

ACTIVE CONDITIONS
{self._format_list('', encounter_conditions['DESCRIPTION'].tolist())}

CURRENT MEDICATIONS
{self._format_list('', encounter_medications['DESCRIPTION'].tolist())}

ONGOING CAREPLANS
{self._format_list('', ongoing_careplans['DESCRIPTION'].tolist())}

NEW CAREPLANS
{self._format_list('', encounter_careplans['DESCRIPTION'].tolist())}
"""
        
        prompt = f"""Task: Simplify this medical note for the patient.

Original Note:
{note}

Please follow these guidelines for the simplified note:
- Write the note in {language}
- Tailor the note to their age
- Use appropriate language for the audience
- Maintain all important medical information
- Explain medical terms when needed
- Include any important instructions or follow-up steps
"""

        return prompt

    def _process_observations(self, observations):
        obs = []
        for i, row in observations.reset_index().iterrows():
            obs.append(f'{row['DESCRIPTION']}: {row['VALUE']} {'' if pd.isna(row['UNITS']) else row['UNITS']}')
        return obs

    def _format_list(self, title: str, items: List[str]) -> str:
        """Format a list of items for the prompt"""
        if not items:
            return f"{title}None"
        return f"{title}" + "\n".join(f"- {item}" for item in items)

    def _format_dict(self, data: Dict) -> str:
        """Format dictionary items for the prompt"""
        return "\n".join(f"- {k}: {v}" for k, v in data.items())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="synthea_sample_data_csv_latest")
    parser.add_argument("--patient", type=str, default="30a6452c-4297-a1ac-977a-6a23237c7b46")
    parser.add_argument("--encounter", type=str, default="0b03e41b-06a6-66fa-b972-acc5a83b134a")
    parser.add_argument("--chain", type=bool, default=False)
    parser.add_argument("--tree", type=bool, default=False)
    parser.add_argument("--few_shot", type=bool, default=False)
    parser.add_argument("--simplification", type=bool, default=False)
    parser.add_argument("--language", type=str, default="English")


    args = parser.parse_args()
    generator = SyntheaPromptGenerator(args.data_path)
    generator.load_data()

    if args.chain == True:
      print("Chain of Thought Prompt:")
      print("-" * 50)
      print(generator.generate_chain_of_thought_prompt(args.encounter))
    
    if args.tree == True:
      print("\nTree of Thoughts Prompt:")
      print("-" * 50)
      print(generator.generate_tree_of_thoughts_prompt(args.encounter))
    
    if args.few_shot == True:
      print("\nFew-Shot Learning Prompt:")
      print("-" * 50)
      print(generator.generate_few_shot_prompt(args.encounter))
    
    if args.simplification == True:
      print("\nMedical Note Simplification Prompt:")
      print("-" * 50)
      print(generator.generate_medical_note_prompt(args.encounter, args.language))