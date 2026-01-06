import os

# ============================================================
# AUTHENTICATION CONFIGURATION
# ============================================================
SERVICE_ACCOUNT_FILE = "vigilant-armor-455313-m8-54c19548a094.json"
PROJECT_ID = "vigilant-armor-455313-m8"
LOCATION = "us-central1"

# ============================================================
# MODEL CONFIGURATION
# ============================================================
MODEL_NAME = "gemini-2.5-pro"
TEMPERATURE = 0.1

# ============================================================
# FILE PATHS
# ============================================================
INPUT_FOLDER = "images_input"
JSON_OUTPUT_FOLDER = "json_output"
EXCEL_OUTPUT_FOLDER = "excel_output"

# Prompt files
POOLED_POPULATION_PROMPT = "prompts/Pooled_Population.txt"
KM_PROMPT = "prompts/KM.txt"
BASELINE_PROMPT = "prompts/Baseline.txt"
RESPONSE_PROMPT = "prompts/RESPONSE.txt"

# ============================================================
# ENSURE DIRECTORIES EXIST
# ============================================================
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(JSON_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(EXCEL_OUTPUT_FOLDER, exist_ok=True)
