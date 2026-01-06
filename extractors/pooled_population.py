import os
import pandas as pd
from pathlib import Path
from pydantic import ValidationError

from schemas import MultiTrialExtractionOutput
from utils import (
    get_genai_client, load_text, load_image_part, safe_json_text,
    save_json, create_output_folders, cleanup_excel_empty_columns
)
from config import POOLED_POPULATION_PROMPT, MODEL_NAME, TEMPERATURE, JSON_OUTPUT_FOLDER, EXCEL_OUTPUT_FOLDER
from logger_config import setup_logger


class PooledPopulationExtractor:
    """Extract pooled population data from poster images."""

    def __init__(self):
        self.logger = setup_logger("PooledPopulation")
        self.client = get_genai_client()
        self.prompt_text = load_text(POOLED_POPULATION_PROMPT)

    def extract(self, image_path: str, image_id: str) -> MultiTrialExtractionOutput:
        """Extract pooled population data from an image."""
        self.logger.info(f"Processing {image_id}")

        folders = create_output_folders(image_id)
        image_part = load_image_part(image_path)

        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=[self.prompt_text, image_part],
            config={
                "response_mime_type": "application/json",
                "response_json_schema": MultiTrialExtractionOutput.model_json_schema(),
                "temperature": TEMPERATURE
            }
        )

        raw_json = safe_json_text(response.text, image_id)

        try:
            parsed = MultiTrialExtractionOutput.model_validate_json(raw_json)
        except ValidationError as e:
            raise RuntimeError(f"Schema validation failed for {image_id}\n{e}")

        json_path = os.path.join(folders["json_folder"], f"{image_id}_pooled_population.json")
        save_json(parsed.model_dump(), json_path)

        excel_path = self._save_excel(parsed, image_id, folders["excel_folder"])
        cleanup_excel_empty_columns(excel_path)

        self.logger.info(f"Saved Excel: {excel_path}")
        self.logger.info(f"Completed {image_id}")
        return parsed

    def _save_excel(self, parsed: MultiTrialExtractionOutput, image_id: str, excel_folder: str):
        """Save extraction results to Excel file with multiple sheets."""
        excel_path = os.path.join(excel_folder, f"{image_id}_pooled_population.xlsx")

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            trial_data = []
            for t in parsed.trial_records:
                record = t.model_dump()

                if 'trial_id_list' in record:
                    if record['trial_id_list'] is not None and isinstance(record['trial_id_list'], list):
                        record['trial_id'] = '; '.join([str(tid) for tid in record['trial_id_list'] if tid])
                    else:
                        record['trial_id'] = ''
                    del record['trial_id_list']

                if 'design_summary' in record:
                    if isinstance(record['design_summary'], dict):
                        record['design_summary'] = record['design_summary'].get('type', '')
                    elif record['design_summary'] is None:
                        record['design_summary'] = ''

                if 'trial_population_details' in record:
                    if isinstance(record['trial_population_details'], dict):
                        record['trial_population_details'] = record['trial_population_details'].get('type', '')
                    elif record['trial_population_details'] is None:
                        record['trial_population_details'] = ''

                trial_data.append(record)

            trial_df = pd.DataFrame(trial_data)
            trial_df.to_excel(writer, sheet_name="trial_records", index=False)

            # Arm records
            arm_df = pd.DataFrame([a.model_dump() for a in parsed.arm_records])
            arm_df.to_excel(writer, sheet_name="arm_records", index=False)

            # Population records
            pop_df = pd.DataFrame([p.model_dump() for p in parsed.population_records])
            pop_df.to_excel(writer, sheet_name="population_records", index=False)

            # Trial arm links
            tal_df = pd.DataFrame([x.model_dump() for x in parsed.trial_arm_links])
            tal_df.to_excel(writer, sheet_name="trial_arm_links", index=False)

            # Trial population links
            tpl_df = pd.DataFrame([x.model_dump() for x in parsed.trial_population_links])
            tpl_df.to_excel(writer, sheet_name="trial_population_links", index=False)

            # Integrated records
            integ_df = pd.DataFrame([x.model_dump() for x in parsed.integrated_records])
            integ_df.to_excel(writer, sheet_name="integrated_records", index=False)

        return excel_path
