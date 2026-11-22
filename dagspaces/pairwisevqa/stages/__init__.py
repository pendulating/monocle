"""Stages for UrbanPairVQA pipeline."""

from .pairwise_vqa import run_pairwise_vqa_stage  # noqa: F401
# trace_extract needs the langextract extra. The orchestrator imports it only
# when the stage runs.

