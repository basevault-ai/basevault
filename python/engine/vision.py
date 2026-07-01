"""
Vision stage — image transcription / description.

Exists as its own module (mirroring extract / entities / patterns /
insights / actions) so `from llm import complete` resolves to the
runner's wrapped `complete` after `_patch_llm_calls`. Per-image
dispatch + the retry cascade (load / sizing-reasoning-off /
other) live in `describe_images_all` — `describe_image` is the
single-attempt leaf that the stage thunk drives.

Vision-specific work: image preprocessing (HEIC/HEIF via the macOS
`sips` tool, BMP/TIFF/… via PIL — all normalized to JPEG),
provider-shape adaptation (OpenAI multimodal vs ollama native), and
per-call output cap plumbing.
"""
from __future__ import annotations

import re
from pathlib import Path

from engine.llm import (
    Mode,
)


# ── Vision model registry + prompt ────────────────────────────────────────


# Vision ship default per mode. LOCAL is DELIBERATELY absent: there is
# no one local vision model id that's correct across both backends
# (`local_backend=mlx` wants `local_mlx_model`, `=ollama` wants
# `local_model`) — the user's actual local model already lives in
# config, and `get_mode_spec(LOCAL)` resolves it per backend. A
# hardcoded LOCAL entry would necessarily pick a tag belonging to one
# backend and break the other. Consumers that read
# `_VISION_MODEL.get(mode, spec.model_id)` (e.g. `runner.py`) fall
# through to the spec's `model_id` for LOCAL on their own; the
# inference path below resolves LOCAL via
# `get_mode_spec(Mode.LOCAL).model_id` directly.
_VISION_MODEL = {
    Mode.TEE: "kimi-k2-6",
}


# Per-call output ceilings. A transcript itself fits well under 4096 (the
# reasoning-off default). With reasoning ON the model spends thousands of
# tokens thinking
# BEFORE the transcript (kimi-k2-6 observed ~5.5k reasoning tokens on a
# single screenshot); a 4096 TOTAL cap truncates mid-thought and returns
# empty visible output, so reasoning-on calls get a far larger ceiling that
# comfortably covers thinking + transcript (still a small fraction of the
# model's window — generation stops once the transcript is done).
_VISION_MAX_OUTPUT = 4096
_VISION_MAX_OUTPUT_REASONING = 32768


_DEFAULT_VISION_PROMPT = """\
Describe the visual content of this image AND transcribe any visible text \
verbatim, preserving the text's layout and structure. Include both when \
both are present. Output only the description when there is no visible \
text. Output only the transcription when the image is purely text (a \
screenshot, document, or chat capture with no separable visual subject) — \
do not fabricate a visual description in that case.

Output ONLY the description and/or transcription itself. Do NOT include \
any preamble such as 'Got it', 'Here is', 'Let me', 'Sure', 'The image \
shows', or any meta-commentary. Start directly with the content.

For screenshots, enumerate the people in the conversation (if visible) at \
the top of the transcript. Each line in the output must have the name of \
a sender and a timestamp.

Be aware of who is the person taking the screenshot (the Author) and who \
is the other person (the Other). In most messaging applications, messages \
from the Author are right-aligned within blue, green, or colored bubbles; \
messages from the Other are left-aligned with grey bubbles.

Explicitly point out quotes — bubbles which respond to previous messages. \
In most cases the Other quotes an older message from the Author, or vice \
versa. If a name for the other person is visible in the screenshot, use \
that name on each line where a message comes from them.

Expected format for screenshots:
Screenshot of a conversation with Alex
[Tue, Jan 30 2020 4:46pm][gray bubble][Alex] Hi!
[Tue, Jan 30 2020 4:47pm][gray bubble][Alex] You still coming?
[Tue, Jan 30 2020 5:15pm][green bubble][Author] Almost! See you soon\
"""


_PREAMBLE_PATTERNS = [
    re.compile(r"^(got it|sure|here(?: is| you go)|let me|okay|ok)[,.!]?\s*.*?[.\n]", re.IGNORECASE),
    re.compile(r"^(the image shows|this image (?:shows|contains|depicts))\s*.*?[.\n]", re.IGNORECASE),
    re.compile(r"^i('ll| will| can)?\s*(transcribe|describe|analyze).*?[.\n]", re.IGNORECASE),
]


def _heic_to_jpeg_bytes(p: Path) -> bytes:
    """Decode a HEIC/HEIF image to JPEG bytes via the macOS built-in
    ``sips`` tool. BaseVault ships as a macOS app, so ``sips`` is always
    present. This keeps HEIC (Apple's default photo format) ingestion
    working without bundling a GPL/LGPL-licensed HEVC codec — pillow-heif's
    wheel embeds GPLv2 ``x265`` even though we only ever decode, which would
    pull GPL into the otherwise-permissive ship."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.jpg"
        subprocess.run(
            ["sips", "-s", "format", "jpeg", str(p), "--out", str(out)],
            check=True, capture_output=True,
        )
        return out.read_bytes()


def encode_image(image_path: str) -> tuple[str, str]:
    """Read an image and return ``(base64, media_type)``. JPG/PNG/WebP/GIF
    pass through; HEIC/HEIF go through macOS ``sips``; everything else
    (BMP/TIFF, …) is converted to JPEG via PIL. Used by the kernel
    INGESTION phase."""
    import base64

    p = Path(image_path)
    suffix = p.suffix.lower()
    native = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif",
    }
    if suffix in native:
        return base64.b64encode(p.read_bytes()).decode("ascii"), native[suffix]
    if suffix in (".heic", ".heif"):
        try:
            return base64.b64encode(_heic_to_jpeg_bytes(p)).decode("ascii"), "image/jpeg"
        except Exception as e:
            raise RuntimeError(f"Cannot read image {p}: {e}") from e
    try:
        import io

        from PIL import Image
        img = Image.open(str(p))
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"
    except Exception as e:
        raise RuntimeError(f"Cannot read image {p}: {e}") from e


def _strip_preamble(text: str) -> str:
    """Best-effort removal of common model preambles from vision output."""
    t = text.lstrip()
    for _ in range(3):  # strip up to 3 layers of preamble
        before = t
        for pat in _PREAMBLE_PATTERNS:
            t = pat.sub("", t, count=1).lstrip()
        if t == before:
            break
    return t.strip()
