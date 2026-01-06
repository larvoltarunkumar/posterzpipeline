import os
import json
import pandas as pd
from typing import Dict, Any
from pathlib import Path

import vertexai
from vertexai.preview.generative_models import GenerativeModel
from google.oauth2 import service_account

from schemas import SurvivalOutput
from utils import (
    load_text, load_json, load_image_for_vertexai, extract_first_json_object,
    build_prompt_with_schema, initialize_vertex_ai, create_output_folders,
    cleanup_excel_empty_columns
)
from config import (
    KM_PROMPT, MODEL_NAME, SERVICE_ACCOUNT_FILE, PROJECT_ID, LOCATION
)
from logger_config import setup_logger


class KMSurvivalExtractor:
    """Extract Kaplan-Meier survival data from poster images."""

    def __init__(self):
        self.logger = setup_logger("KMSurvival")
        self.credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE
        )
        vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=self.credentials)
        self.model = GenerativeModel(MODEL_NAME)
        self.prompt_txt = load_text(KM_PROMPT)

    def extract(
        self,
        image_path: str,
        image_id: str,
        pooled_json_path: str
    ) -> SurvivalOutput:
        """Extract KM survival data from image using pooled population schema."""
        self.logger.info(f"Processing {image_id}")

        folders = create_output_folders(image_id)

        input2_schema = load_json(pooled_json_path)
        final_prompt = build_prompt_with_schema(self.prompt_txt, input2_schema)
        image = load_image_for_vertexai(image_path)

        response = self.model.generate_content(
            contents=[final_prompt, image],
            generation_config={"temperature": 0.0}
        )

        raw_text = response.text or ""
        parsed = extract_first_json_object(raw_text)
        filtered = self._clean_to_survival_only(parsed)
        validated = SurvivalOutput.model_validate(filtered, by_name=True)

        output_json_path = Path(folders["json_folder"]) / f"{image_id}_km_survival.json"
        output_json_path.write_text(
            validated.model_dump_json(indent=2),
            encoding="utf-8"
        )

        output_excel_path = Path(folders["excel_folder"]) / f"{image_id}_km_survival.xlsx"
        self._save_excel(validated, str(output_excel_path))
        cleanup_excel_empty_columns(str(output_excel_path))

        self.logger.info(f"Saved Excel: {output_excel_path}")
        self.logger.info(f"Completed {image_id}")
        return validated

    def _clean_to_survival_only(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only trial_metadata and arm_level_survival_outcomes."""
        ALLOWED_TM_KEYS = {"trial_id", "phase", "study_name"}
        ALLOWED_OUTCOME_KEYS = {
            "survival_outcome_id", "trial_id", "trial_label", "arm_description",
            "population_type", "population_description", "endpoint_description",
            "endpoint_name", "endpoint_label", "assessment_type", "review_board",
            "review_criteria", "other_details", "arm_n", "median_survival",
            "survival_rate", "events_n", "assessment_denominator_n",
            "p_value", "time_unit"
        }

        tm_raw = parsed.get("trial_metadata", {}) or {}
        rows_raw = parsed.get("arm_level_survival_outcomes", []) or []

        tm = {k: tm_raw.get(k) for k in ALLOWED_TM_KEYS if k in tm_raw}

        rows = rows_raw if isinstance(rows_raw, list) else []
        cleaned_rows = []
        for r in rows:
            if isinstance(r, dict):
                cleaned_rows.append({k: r.get(k) for k in ALLOWED_OUTCOME_KEYS if k in r})

        return {"trial_metadata": tm, "arm_level_survival_outcomes": cleaned_rows}

    def _save_excel(self, validated: SurvivalOutput, excel_path: str):
        """Export survival data to Excel with flattened structure."""
        data = json.loads(validated.model_dump_json())
        trial_metadata = data.get("trial_metadata", {}) or {}
        arm_outcomes = data.get("arm_level_survival_outcomes", []) or []

        rows = []
        for row in arm_outcomes:
            if isinstance(row, dict):
                merged = {**trial_metadata, **row}
                rows.append(merged)

        df = pd.DataFrame(rows)
        df.to_excel(excel_path, index=False)
