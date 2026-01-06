import os
import json
import pandas as pd
from typing import Dict, Any, List
from pathlib import Path

import vertexai
from vertexai.preview.generative_models import GenerativeModel
from google.oauth2 import service_account

from schemas import ResponseOutput
from utils import (
    load_text, load_json, load_image_for_vertexai, extract_first_json_object,
    build_prompt_with_schema, create_output_folders, cleanup_excel_empty_columns
)
from config import (
    RESPONSE_PROMPT, MODEL_NAME, SERVICE_ACCOUNT_FILE, PROJECT_ID, LOCATION
)
from logger_config import setup_logger


class ResponseOutcomesExtractor:
    """Extract response outcomes from poster images."""

    def __init__(self):
        self.logger = setup_logger("ResponseOutcomes")
        self.credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE
        )
        vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=self.credentials)
        self.model = GenerativeModel(MODEL_NAME)
        self.prompt_txt = load_text(RESPONSE_PROMPT)

    def extract(
        self,
        image_path: str,
        image_id: str,
        pooled_json_path: str
    ) -> ResponseOutput:
        """Extract response outcomes from image using pooled population schema."""
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
        filtered = self._clean_to_response_only(parsed)
        validated = ResponseOutput.model_validate(filtered)

        output_json_path = Path(folders["json_folder"]) / f"{image_id}_response_outcomes.json"
        output_json_path.write_text(
            validated.model_dump_json(indent=2),
            encoding="utf-8"
        )

        output_excel_path = Path(folders["excel_folder"]) / f"{image_id}_response_outcomes.xlsx"
        self._save_excel(validated, str(output_excel_path))
        cleanup_excel_empty_columns(str(output_excel_path))

        self.logger.info(f"Saved Excel: {output_excel_path}")
        self.logger.info(f"Completed {image_id}")
        return validated

    def _clean_to_response_only(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only trial_metadata and arm_level_response_outcomes."""
        ALLOWED_TM_KEYS = {"trial_id", "phase", "study_name"}
        ALLOWED_OUTCOME_KEYS = {
            "response_outcome_id", "trial_id", "trial_label", "arm_description",
            "population_type", "population_description", "assessment_type",
            "review_board", "review_criteria", "other_details", "arm_n",
            "assessment_denominator_n", "response_type_name",
            "response_metric_class", "result"
        }

        tm_raw = parsed.get("trial_metadata", {}) or {}
        rows_raw = parsed.get("arm_level_response_outcomes", []) or []

        tm = {k: tm_raw.get(k) for k in ALLOWED_TM_KEYS if k in tm_raw}

        rows = rows_raw if isinstance(rows_raw, list) else []
        cleaned_rows = []
        for r in rows:
            if isinstance(r, dict):
                cleaned_rows.append({k: r.get(k) for k in ALLOWED_OUTCOME_KEYS if k in r})

        return {
            "trial_metadata": tm,
            "arm_level_response_outcomes": cleaned_rows
        }

    def _save_excel(self, validated: ResponseOutput, excel_path: str):
        """Export response data to Excel with flattened result columns."""
        data = json.loads(validated.model_dump_json())
        trial_metadata = data.get("trial_metadata", {}) or {}
        arm_outcomes = data.get("arm_level_response_outcomes", []) or []

        flattened_rows = []

        for row in arm_outcomes:
            if not isinstance(row, dict):
                continue

            result = row.pop("result", {}) or {}

            flattened = {
                **trial_metadata,
                **row,
                # Flatten result fields
                "result_n": result.get("n"),
                "result_percentage": result.get("percentage"),
                "result_min": result.get("min"),
                "result_max": result.get("max"),
                "result_p_value": result.get("p_value"),
                "result_odds_ratio": result.get("odds_ratio"),
                "result_median": result.get("median"),
                "result_min_duration": result.get("min_duration"),
                "result_max_duration": result.get("max_duration"),
                "result_duration_unit": result.get("duration_unit")
            }

            flattened_rows.append(flattened)

        df = pd.DataFrame(flattened_rows)
        df.to_excel(excel_path, index=False)
