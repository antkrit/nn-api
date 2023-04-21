"""Contains global config object."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASEDIR = Path(__file__).resolve(strict=True).parent.parent
dotenv_path = BASEDIR / ".env"

load_dotenv(dotenv_path)

config = os.environ
