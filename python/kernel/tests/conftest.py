from pathlib import Path

from dotenv import load_dotenv

# Reuse the ONE canonical key prod uses — no second copy, no new secret.
load_dotenv(Path.home() / "Library" / "Application Support" / "BaseVault" / ".env")
