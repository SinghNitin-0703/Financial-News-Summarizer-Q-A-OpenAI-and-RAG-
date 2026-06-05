import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve .env relative to project root (parent of src/), not CWD
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / "OpenAI_APIkey.env")


class Config:
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-02-01")
    FAISS_INDEX_PATH = str(_project_root / "faiss_index")


def validate_config():
    """Raise early if required Azure credentials are missing."""
    missing = [
        var for var in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY")
        if not getattr(Config, var)
    ]
    if missing:
        raise EnvironmentError(
            f"Missing required env vars: {', '.join(missing)}. "
            f"Create 'OpenAI_APIkey.env' in the project root."
        )