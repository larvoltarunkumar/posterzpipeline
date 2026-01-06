import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from extractors import (
    PooledPopulationExtractor,
    KMSurvivalExtractor,
    BaselineExtractor,
    ResponseOutcomesExtractor
)
from config import INPUT_FOLDER
from logger_config import setup_logger


class PosterPipeline:
    """Pipeline orchestrator for poster data extraction."""

    def __init__(self):
        self.logger = setup_logger("Pipeline")
        self.pooled_extractor = PooledPopulationExtractor()
        self.km_extractor = KMSurvivalExtractor()
        self.baseline_extractor = BaselineExtractor()
        self.response_extractor = ResponseOutcomesExtractor()

    def process_image(self, image_path: str) -> dict:
        """Process a single image through the complete pipeline."""
        image_id = Path(image_path).stem
        self.logger.info("="*60)
        self.logger.info(f"Processing: {image_id}")
        self.logger.info("="*60)

        results = {
            "image_id": image_id,
            "image_path": image_path,
            "pooled_population": None,
            "km_survival": None,
            "baseline": None,
            "response_outcomes": None
        }

        self.logger.info("[STAGE 1] Pooled Population Extraction")
        pooled_result = self.pooled_extractor.extract(image_path, image_id)
        results["pooled_population"] = pooled_result

        pooled_json_path = f"json_output/{image_id}/{image_id}_pooled_population.json"

        self.logger.info("[STAGE 2] Parallel Extraction (KM, Baseline, Response)")
        parallel_results = self._run_parallel_extractions(
            image_path, image_id, pooled_json_path
        )

        results.update(parallel_results)

        self.logger.info("="*60)
        self.logger.info(f"Completed processing: {image_id}")
        self.logger.info("="*60)

        return results

    def _run_parallel_extractions(
        self,
        image_path: str,
        image_id: str,
        pooled_json_path: str
    ) -> dict:
        """Run KM, Baseline, and Response extractions in parallel."""
        results = {
            "km_survival": None,
            "baseline": None,
            "response_outcomes": None
        }

        tasks = [
            ("km_survival", self.km_extractor.extract, (image_path, image_id, pooled_json_path)),
            ("baseline", self.baseline_extractor.extract, (image_path, image_id, pooled_json_path)),
            ("response_outcomes", self.response_extractor.extract, (image_path, image_id, pooled_json_path))
        ]

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_task = {
                executor.submit(task_func, *task_args): task_name
                for task_name, task_func, task_args in tasks
            }

            for future in as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    result = future.result()
                    results[task_name] = result
                    self.logger.info(f"[Parallel] {task_name} completed")
                except Exception as e:
                    self.logger.error(f"[Parallel] {task_name} failed: {e}")
                    results[task_name] = None

        return results

    def _ai_jury_verification(self, results: dict) -> dict:
        """Future implementation: AI Jury verification layer."""
        self.logger.info("[AI Jury] Verification not yet implemented")
        return results

    def process_all_images(self) -> List[dict]:
        """Process all images in the input folder."""
        image_files = [
            f for f in os.listdir(INPUT_FOLDER)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]

        if not image_files:
            self.logger.warning(f"No images found in {INPUT_FOLDER}")
            return []

        self.logger.info(f"Found {len(image_files)} image(s) to process")

        all_results = []
        for image_file in image_files:
            image_path = os.path.join(INPUT_FOLDER, image_file)
            try:
                results = self.process_image(image_path)
                all_results.append(results)
            except Exception as e:
                self.logger.error(f"Failed to process {image_file}: {e}")
                all_results.append({
                    "image_id": Path(image_file).stem,
                    "image_path": image_path,
                    "error": str(e)
                })

        return all_results
