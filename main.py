"""
Pipeline Flow:
1. Pooled Population Extraction (from image)
2. Parallel Extraction:
   - KM Survival Analysis
   - Baseline Characteristics
   - Response Outcomes
3. [Future] AI Jury Verification

Usage:
    python main.py
"""

import sys
from pipeline import PosterPipeline


def main():
    """Main entry point for the pipeline."""
    print("\n" + "="*60)
    print("Poster Data Extraction Pipeline")
    print("="*60)

    try:
        # Initialize and run pipeline
        pipeline = PosterPipeline()
        results = pipeline.process_all_images()

        # Summary
        print("\n" + "="*60)
        print("PIPELINE SUMMARY")
        print("="*60)
        print(f"Total images processed: {len(results)}")

        successful = sum(1 for r in results if "error" not in r)
        failed = len(results) - successful

        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

        if failed > 0:
            print("\nFailed images:")
            for r in results:
                if "error" in r:
                    print(f"  - {r['image_id']}: {r['error']}")

        print("="*60 + "\n")

        return 0 if failed == 0 else 1

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
