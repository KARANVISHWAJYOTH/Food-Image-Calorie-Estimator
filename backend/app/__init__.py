"""
NutriVision AI - Food Image Calorie Estimator Backend
Powered by FastAPI, PyTorch & Multi-Task CNN
"""
from pathlib import Path

from dotenv import load_dotenv

# Load repository-root .env before any submodule reads configuration.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

__version__ = "1.0.0"
