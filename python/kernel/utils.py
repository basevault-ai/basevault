from __future__ import annotations

import datetime
import secrets
from dataclasses import dataclass

# Skip 1, 0, i, l letters which are easy to confuse between each other.
_SHORT_ID_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"


@dataclass
class PermaId:
    short_id: str
    full_id: str

    @staticmethod
    def new() -> PermaId:
        """Generate a pseudo-unique id by concatenating the current timestamp with a 4 char random string."""
        short_id = "".join(secrets.choice(_SHORT_ID_ALPHABET) for _ in range(4))
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        return PermaId(short_id, f"{ts}-{short_id}")
