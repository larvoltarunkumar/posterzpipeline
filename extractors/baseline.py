import os
import json
import re
import pandas as pd
from typing import Dict, Any, List
from pathlib import Path

import vertexai
from vertexai.preview.generative_models import GenerativeModel
from google.oauth2 import service_account

from schemas import BaselineOutput
from utils import (
    load_text, load_json, load_image_for_vertexai, extract_first_json_object,
    build_prompt_with_schema, to_int_or_none, to_float_or_none,
    create_output_folders, cleanup_excel_empty_columns
)
from config import (
    BASELINE_PROMPT, MODEL_NAME, SERVICE_ACCOUNT_FILE, PROJECT_ID, LOCATION
)
from logger_config import setup_logger


class BaselineExtractor:
    """Extract baseline characteristics from poster images."""

    def __init__(self):
        self.logger = setup_logger("Baseline")
        self.credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE
        )
        vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=self.credentials)
        self.model = GenerativeModel(MODEL_NAME)
        self.prompt_txt = load_text(BASELINE_PROMPT)

    def extract(
        self,
        image_path: str,
        image_id: str,
        pooled_json_path: str
    ) -> BaselineOutput:
        """Extract baseline characteristics from image using pooled population schema."""
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
        filtered = self._clean_to_bc_only(parsed)
        validated = BaselineOutput.model_validate(filtered, by_name=True)

        output_json_path = Path(folders["json_folder"]) / f"{image_id}_baseline.json"
        output_json_path.write_text(
            validated.model_dump_json(indent=2, exclude_none=False),
            encoding="utf-8"
        )

        output_excel_path = Path(folders["excel_folder"]) / f"{image_id}_baseline.xlsx"
        self._save_excel(validated, str(output_excel_path))
        cleanup_excel_empty_columns(str(output_excel_path))

        self.logger.info(f"Saved Excel: {output_excel_path}")
        self.logger.info(f"Completed {image_id}")
        return validated

    def _clean_to_bc_only(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only bc_types and normalize data."""
        ALLOWED_BC_KEYS = {
            "baseline_id", "trial_id", "trial_label", "arm_key", "arm_description",
            "population_key", "population_type", "population_description",
            "baseline_parent", "parent_description", "baseline_category_label",
            "group_label", "group_text", "measure", "measure_value",
            "population_n", "population_percentage"
        }

        bc_raw = parsed.get("bc_types", [])
        rows = bc_raw if isinstance(bc_raw, list) else []

        cleaned_rows: List[Dict[str, Any]] = []
        for r in rows:
            if not isinstance(r, dict):
                continue

            rr = {k: r.get(k) for k in ALLOWED_BC_KEYS if k in r}

            # Normalize enums
            rr["population_type"] = self._normalize_population_type(rr.get("population_type"))
            rr["baseline_parent"] = self._normalize_baseline_parent(rr.get("baseline_parent"))

            # Normalize numeric fields
            rr["population_n"] = to_int_or_none(rr.get("population_n"))
            rr["population_percentage"] = to_float_or_none(rr.get("population_percentage"))

            cleaned_rows.append(rr)

        # Fill baseline_id if missing
        for i, rr in enumerate(cleaned_rows, start=1):
            bid = rr.get("baseline_id")
            if not isinstance(bid, int) or bid <= 0:
                rr["baseline_id"] = i

        return {"bc_types": cleaned_rows}

    def _normalize_population_type(self, v: Any) -> str:
        """Normalize population_type to allowed values."""
        allowed = {"Overall", "Analysis set", "Cohort", "Subgroup", "Other"}
        if isinstance(v, str) and v.strip() in allowed:
            return v.strip()
        return "Other"

    def _normalize_baseline_parent(self, v: Any) -> str:
        """Normalize baseline_parent to allowed values."""
        allowed = {"Overall", "Cohort", "Subgroup", "Other"}
        if v is None:
            return None
        if isinstance(v, str) and v.strip() in allowed:
            return v.strip()
        return None

    def _save_excel(self, validated: BaselineOutput, excel_path: str):
        """Export baseline data to Excel."""
        data = json.loads(validated.model_dump_json())
        bc_rows = data.get("bc_types", []) or []

        df = pd.DataFrame(bc_rows)

        # Reorder columns
        preferred_cols = [
            "baseline_id", "trial_id", "trial_label", "arm_key", "arm_description",
            "population_key", "population_type", "population_description",
            "baseline_parent", "parent_description", "baseline_category_label",
            "group_label", "group_text", "measure", "measure_value",
            "population_n", "population_percentage"
        ]
        df = df.reindex(columns=[c for c in preferred_cols if c in df.columns])

        df.to_excel(excel_path, index=False)
