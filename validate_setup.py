"""
Setup Validation Script

Run this script to validate your environment setup before running the pipeline.
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """Check Python version is 3.12+."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 12:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python version: {version.major}.{version.minor}.{version.micro}")
        print("   Required: Python 3.12+")
        return False


def check_required_modules():
    """Check if all required modules are installed."""
    required_modules = [
        "google.genai",
        "vertexai",
        "google.oauth2",
        "pandas",
        "openpyxl",
        "pydantic"
    ]

    all_ok = True
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ Module installed: {module}")
        except ImportError:
            print(f"❌ Module missing: {module}")
            all_ok = False

    return all_ok


def check_credentials():
    """Check if credentials file exists."""
    cred_file = "vigilant-armor-455313-m8-1d642ef84a8c.json"
    if os.path.exists(cred_file):
        print(f"✅ Credentials file found: {cred_file}")
        return True
    else:
        print(f"❌ Credentials file not found: {cred_file}")
        return False


def check_directories():
    """Check if required directories exist."""
    required_dirs = [
        "images_input",
        "json_output",
        "excel_output",
        "extractors"
    ]

    all_ok = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ Directory exists: {directory}/")
        else:
            print(f"❌ Directory missing: {directory}/")
            all_ok = False

    return all_ok


def check_prompt_files():
    """Check if all prompt files exist."""
    prompt_files = [
        "Pooled_Population.txt",
        "KM.txt",
        "Baseline.txt",
        "RESPONSE.txt"
    ]

    all_ok = True
    for prompt_file in prompt_files:
        if os.path.exists(prompt_file):
            print(f"✅ Prompt file exists: {prompt_file}")
        else:
            print(f"⚠️  Prompt file missing: {prompt_file}")
            all_ok = False

    return all_ok


def check_core_files():
    """Check if core Python files exist."""
    core_files = [
        "main.py",
        "pipeline.py",
        "config.py",
        "schemas.py",
        "utils.py"
    ]

    all_ok = True
    for core_file in core_files:
        if os.path.exists(core_file):
            print(f"✅ Core file exists: {core_file}")
        else:
            print(f"❌ Core file missing: {core_file}")
            all_ok = False

    return all_ok


def main():
    """Run all validation checks."""
    print("="*60)
    print("Pipeline Setup Validation")
    print("="*60)

    checks = [
        ("Python Version", check_python_version),
        ("Required Modules", check_required_modules),
        ("Credentials", check_credentials),
        ("Directories", check_directories),
        ("Prompt Files", check_prompt_files),
        ("Core Files", check_core_files)
    ]

    print("\n" + "="*60)
    results = []

    for check_name, check_func in checks:
        print(f"\n[{check_name}]")
        result = check_func()
        results.append((check_name, result))

    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)

    all_passed = all(result for _, result in results)

    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {check_name}")

    print("="*60)

    if all_passed:
        print("\n✅ All checks passed! You're ready to run the pipeline.")
        print("   Run: python main.py")
        return 0
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Missing modules: pip install -r requirements.txt")
        print("  - Missing directories: they will be created on first run")
        print("  - Missing credentials: ensure service account JSON is present")
        return 1


if __name__ == "__main__":
    sys.exit(main())
