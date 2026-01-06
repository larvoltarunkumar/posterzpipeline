import json
import re
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import vertexai
from google import genai
from google.genai import types
from google.oauth2 import service_account

from config import SERVICE_ACCOUNT_FILE, PROJECT_ID, LOCATION
from logger_config import setup_logger

logger = setup_logger("Utils")

def get_credentials():
    """Get Google Cloud credentials from service account file."""
    return service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE
    ).with_scopes(["https://www.googleapis.com/auth/cloud-platform"])


def initialize_vertex_ai():
    """Initialize Vertex AI with credentials."""
    credentials = get_credentials()
    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
    return credentials


def get_genai_client():
    """Get Google GenAI client."""
    credentials = get_credentials()
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
        credentials=credentials
    )
    return client


# ============================================================
# FILE UTILITIES
# ============================================================

def load_text(file_path: str) -> str:
    """Load text content from a file."""
    return Path(file_path).read_text(encoding="utf-8")


def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON content from a file."""
    return json.loads(Path(file_path).read_text(encoding="utf-8"))


def save_json(data: Any, file_path: str):
    """Save data as JSON to a file."""
    Path(file_path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def load_image_part(image_path: str) -> types.Part:
    """Load image and create GenAI Part object for google.genai client."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    ext = Path(image_path).suffix.lower()
    mime = "image/jpeg"
    if ext == ".png":
        mime = "image/png"
    elif ext == ".webp":
        mime = "image/webp"

    return types.Part.from_bytes(data=image_bytes, mime_type=mime)


def load_image_for_vertexai(image_path: str):
    """Load image for Vertex AI GenerativeModel (returns raw image data)."""
    from vertexai.preview.generative_models import Part, Image

    ext = Path(image_path).suffix.lower()
    mime = "image/jpeg"
    if ext == ".png":
        mime = "image/png"
    elif ext == ".webp":
        mime = "image/webp"

    # Load image using Vertex AI's Image class
    image = Image.load_from_file(image_path)
    return image


# ============================================================
# TEXT PROCESSING
# ============================================================

def safe_json_text(model_text: str, context: str = "") -> str:
    """Validate that model output is valid JSON."""
    txt = (model_text or "").strip()
    if not (txt.startswith("{") and txt.endswith("}")):
        raise RuntimeError(
            f"Model output not valid JSON / truncated {context}:\n{txt[:500]}"
        )
    return txt


def extract_first_json_object(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from text (handles markdown fences)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model output text.")

    # Remove markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # If pure JSON already
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    # Try to find first {...}
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output.")
    return json.loads(match.group(0))


def build_prompt_with_schema(prompt_txt: str, schema_json: Dict[str, Any]) -> str:
    """Replace {INPUT_2_SCHEMA_JSON} placeholder with actual schema."""
    schema_str = json.dumps(schema_json, ensure_ascii=False, indent=2)
    if "{INPUT_2_SCHEMA_JSON}" in prompt_txt:
        return prompt_txt.replace("{INPUT_2_SCHEMA_JSON}", schema_str)
    return prompt_txt + "\n\nINPUT 2 JSON schema reference:\n" + schema_str


# ============================================================
# DATA NORMALIZATION UTILITIES
# ============================================================

def to_int_or_none(x: Any) -> Optional[int]:
    """Convert value to int or None."""
    if x is None:
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        m = re.search(r"-?\d+", s)
        if not m:
            return None
        try:
            return int(m.group(0))
        except Exception:
            return None
    return None


def to_float_or_none(x: Any) -> Optional[float]:
    """Convert value to float or None."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        s = s.replace("%", "").strip()
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:
            return None
    return None


# ============================================================
# EXCEL UTILITIES
# ============================================================

def cleanup_excel_empty_columns(excel_path: str):
    """
    Remove columns that have no data (all None/NaN/empty) from an Excel file.

    Args:
        excel_path: Path to the Excel file to clean up
    """
    if not Path(excel_path).exists():
        return

    # Read the Excel file (handle multi-sheet if needed)
    excel_file = pd.ExcelFile(excel_path)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)

            # Remove columns where all values are NaN/None/empty
            df = df.dropna(axis=1, how='all')

            # Also remove columns where all values are empty strings
            for col in df.columns:
                if df[col].astype(str).str.strip().eq('').all():
                    df = df.drop(columns=[col])

            # Write cleaned dataframe back
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def create_output_folders(image_id: str) -> dict:
    """
    Create output folder structure for a given image_id.

    Returns dict with paths:
    - excel_folder: excel_output/{image_id}/
    - json_folder: json_output/{image_id}/
    """
    from config import EXCEL_OUTPUT_FOLDER, JSON_OUTPUT_FOLDER

    excel_folder = Path(EXCEL_OUTPUT_FOLDER) / image_id
    json_folder = Path(JSON_OUTPUT_FOLDER) / image_id

    excel_folder.mkdir(parents=True, exist_ok=True)
    json_folder.mkdir(parents=True, exist_ok=True)

    return {
        "excel_folder": str(excel_folder),
        "json_folder": str(json_folder)
    }


# def consolidate_outputs(image_id: str):
#     """Consolidate all individual JSON and Excel outputs into final combined files."""
#     from config import EXCEL_OUTPUT_FOLDER, JSON_OUTPUT_FOLDER

#     excel_folder = Path(EXCEL_OUTPUT_FOLDER) / image_id
#     json_folder = Path(JSON_OUTPUT_FOLDER) / image_id

#     pooled_json_path = json_folder / f"{image_id}_pooled_population.json"

#     if pooled_json_path.exists():
#         pooled_data = load_json(str(pooled_json_path))

#         consolidated_json = {
#             "trial_records": pooled_data.get("trial_records", []),
#             "arm_records": pooled_data.get("arm_records", []),
#             "KM": None,
#             "Baseline": None,
#             "Response": None
#         }
#     else:
#         consolidated_json = {
#             "trial_records": [],
#             "arm_records": [],
#             "KM": None,
#             "Baseline": None,
#             "Response": None
#         }

#     km_json_path = json_folder / f"{image_id}_km_survival.json"
#     if km_json_path.exists():
#         consolidated_json["KM"] = load_json(str(km_json_path))

#     baseline_json_path = json_folder / f"{image_id}_baseline.json"
#     if baseline_json_path.exists():
#         consolidated_json["Baseline"] = load_json(str(baseline_json_path))

#     response_json_path = json_folder / f"{image_id}_response_outcomes.json"
#     if response_json_path.exists():
#         consolidated_json["Response"] = load_json(str(response_json_path))

#     consolidated_json_path = json_folder / f"{image_id}.json"
#     save_json(consolidated_json, str(consolidated_json_path))
#     logger.info(f"Saved consolidated JSON: {consolidated_json_path}")

#     consolidated_excel_path = excel_folder / f"{image_id}.xlsx"
#     sheet_count = 0

#     with pd.ExcelWriter(str(consolidated_excel_path), engine="openpyxl") as writer:
#         if consolidated_json["trial_records"]:
#             trial_df = pd.json_normalize(consolidated_json["trial_records"])
#             trial_df.to_excel(writer, sheet_name="trial_records", index=False)
#             sheet_count += 1

#         if consolidated_json["arm_records"]:
#             arm_df = pd.DataFrame(consolidated_json["arm_records"])
#             arm_df.to_excel(writer, sheet_name="arm_records", index=False)
#             sheet_count += 1

#         if consolidated_json["KM"]:
#             km_data = consolidated_json["KM"]
#             trial_metadata = km_data.get("trial_metadata", {}) or {}
#             arm_outcomes = km_data.get("arm_level_survival_outcomes", []) or []

#             rows = []
#             for row in arm_outcomes:
#                 if isinstance(row, dict):
#                     merged = {**trial_metadata, **row}
#                     rows.append(merged)

#             if rows:
#                 km_df = pd.DataFrame(rows)
#                 km_df.to_excel(writer, sheet_name="KM", index=False)
#                 sheet_count += 1

#         if consolidated_json["Baseline"]:
#             baseline_data = consolidated_json["Baseline"]
#             bc_rows = baseline_data.get("bc_types", []) or []

#             if bc_rows:
#                 baseline_df = pd.DataFrame(bc_rows)
#                 baseline_df.to_excel(writer, sheet_name="Baseline", index=False)
#                 sheet_count += 1

#         if consolidated_json["Response"]:
#             response_data = consolidated_json["Response"]
#             trial_metadata = response_data.get("trial_metadata", {}) or {}
#             arm_outcomes = response_data.get("arm_level_response_outcomes", []) or []

#             flattened_rows = []
#             for row in arm_outcomes:
#                 if not isinstance(row, dict):
#                     continue

#                 result = row.pop("result", {}) or {}
#                 flattened = {
#                     **trial_metadata,
#                     **row,
#                     "result_n": result.get("n"),
#                     "result_percentage": result.get("percentage"),
#                     "result_min": result.get("min"),
#                     "result_max": result.get("max"),
#                     "result_p_value": result.get("p_value"),
#                     "result_odds_ratio": result.get("odds_ratio"),
#                     "result_median": result.get("median"),
#                     "result_min_duration": result.get("min_duration"),
#                     "result_max_duration": result.get("max_duration"),
#                     "result_duration_unit": result.get("duration_unit")
#                 }
#                 flattened_rows.append(flattened)

#             if flattened_rows:
#                 response_df = pd.DataFrame(flattened_rows)
#                 response_df.to_excel(writer, sheet_name="Response", index=False)
#                 sheet_count += 1

#         if sheet_count == 0:
#             pd.DataFrame({"info": ["No data available"]}).to_excel(writer, sheet_name="Summary", index=False)
#             logger.warning(f"No data found for {image_id}, created summary sheet")

#     cleanup_excel_empty_columns(str(consolidated_excel_path))
#     logger.info(f"Saved consolidated Excel with {sheet_count} sheet(s): {consolidated_excel_path}")

#     return {
#         "json_path": str(consolidated_json_path),
#         "excel_path": str(consolidated_excel_path)
#     }
