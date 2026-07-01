from __future__ import annotations

import os
from enum import Enum


class Environment(Enum):
    PROD = 1
    DEV = 2

    @staticmethod
    def get() -> Environment:
        if os.environ.get("BASEVAULT_DEV"):
            return Environment.DEV
        return Environment.PROD


class JobName(Enum):
    FULL_PIPELINE = "Full Pipeline"
    EMBEDDING_ONLY = "Embedding Only"


class StageName(Enum):
    INGESTION = "ingestion"
    EXTRACTION = "extraction_splitter"
    ENTITIES = "entities"
    PATTERNS = "patterns"
    INSIGHTS = "insights"
    ACTIONS = "actions"
    EMBEDDINGS = "embeddings"
    CHAT = "chat"


class PhaseName(Enum):
    INGESTION = "ingestion"
    EXTRACTION_SPLITTER = "extraction_splitter"
    EXTRACTION_LLM = "extraction_llm"
    EXTRACTION_COMPLETION = "extraction_completion"
    ENTITY_GROUPING = "entity_grouping"
    ENTITY_SUMMARIZE = "entity_summarize"
    ENTITY_DEDUPE = "entity_dedupe"
    PATTERNS = "patterns"
    INSIGHTS = "insights"
    ACTIONS = "actions"
    EMBEDDINGS = "embeddings"
    CHAT = "chat"

    def does_llm_call(self) -> bool:
        return self in {
            PhaseName.INGESTION,
            PhaseName.EXTRACTION_LLM,
            PhaseName.ENTITY_SUMMARIZE,
            PhaseName.ENTITY_DEDUPE,
            PhaseName.PATTERNS,
            PhaseName.INSIGHTS,
            PhaseName.ACTIONS,
            PhaseName.EMBEDDINGS,
            PhaseName.CHAT,
        }

    def is_non_degrading(self) -> bool:
        return self in {
            PhaseName.EXTRACTION_LLM,
            PhaseName.ENTITY_SUMMARIZE,
            PhaseName.EMBEDDINGS,
        }

    def is_degrading(self) -> bool:
        return self in {
            PhaseName.INGESTION,
            PhaseName.ENTITY_DEDUPE,
            PhaseName.PATTERNS,
            PhaseName.INSIGHTS,
            PhaseName.ACTIONS,
            PhaseName.CHAT,
        }

    def stage_name(self) -> StageName:
        if self == PhaseName.INGESTION:
            return StageName.INGESTION
        elif self in {
            PhaseName.EXTRACTION_SPLITTER,
            PhaseName.EXTRACTION_LLM,
            PhaseName.EXTRACTION_COMPLETION,
        }:
            return StageName.EXTRACTION
        elif self in {
            PhaseName.ENTITY_GROUPING,
            PhaseName.ENTITY_SUMMARIZE,
            PhaseName.ENTITY_DEDUPE,
        }:
            return StageName.ENTITIES
        elif self == PhaseName.PATTERNS:
            return StageName.PATTERNS
        elif self == PhaseName.INSIGHTS:
            return StageName.INSIGHTS
        elif self == PhaseName.ACTIONS:
            return StageName.ACTIONS
        elif self == PhaseName.EMBEDDINGS:
            return StageName.EMBEDDINGS
        elif self == PhaseName.CHAT:
            return StageName.CHAT
        raise ValueError(f"Unknown phase: {self}")


class LlmStatus(Enum):
    LOAD = "error (load)"
    OTHER = "error (other)"
    CAP_HIT = "cap hit (sizing)"
    PARSE_ERROR = "parse error (sizing)"
    TIMEOUT_WITH_TOKENS = "timeout (sizing)"
    ABORTED = "aborted"
    SKIPPED = "skipped"
    OK = "ok"
    SUCCESS_EMPTY = "success empty"
    SUCCESS_SAMPLED = "success sampled"
    SUCCESS_REASONING_OFF = "success reasoning off"
    SUCCESS_MODEL_FALLBACK = "success model fallback"


class RetryType(Enum):
    NO_RETRY = "no retry"
    FULL_RETRY = "full retry"
    HALVES = "halves"
    SAMPLE = "sample"
    REASONING_OFF = "reasoning off"
    MODEL_FALLBACK = "model fallback"


class AttestationType(Enum):
    INTEL_TDX = ("intel-tdx", [0x1A8, 0x1D8], 0x238, 0x258)  # RTMR1 and RTMR2.
    AMD_SEV_SNP = ("amd-sev-snp", [0x90], 0x50, 0x70)

    def __init__(
        self,
        name: str,
        measurement_offsets: list[int],
        tls_pubkey_fingerprint_offset: int,
        hpke_pubkey_offset: int,
    ):
        self._name_ = name
        self.measurement_offsets = measurement_offsets
        self.tls_pubkey_fingerprint_offset = tls_pubkey_fingerprint_offset
        self.hpke_pubkey_offset = hpke_pubkey_offset
