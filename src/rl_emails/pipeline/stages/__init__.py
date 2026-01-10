"""Pipeline stages module.

Each stage is a module that exports a run() function returning StageResult.
"""

from __future__ import annotations

from rl_emails.pipeline.stages import (
    stage_01_parse_mbox,
    stage_02_import_postgres,
    stage_03_populate_threads,
    stage_04_enrich_emails,
    stage_05_compute_features,
    stage_06_compute_embeddings,
    stage_07_classify_handleability,
    stage_08_populate_users,
    stage_09_cluster_emails,
    stage_10_compute_priority,
    stage_11_llm_classification,
    stage_12_entity_extraction,
    stage_13_enhance_clusters,
)
from rl_emails.pipeline.stages.base import StageResult

__all__ = [
    "StageResult",
    "stage_01_parse_mbox",
    "stage_02_import_postgres",
    "stage_03_populate_threads",
    "stage_04_enrich_emails",
    "stage_05_compute_features",
    "stage_06_compute_embeddings",
    "stage_07_classify_handleability",
    "stage_08_populate_users",
    "stage_09_cluster_emails",
    "stage_10_compute_priority",
    "stage_11_llm_classification",
    "stage_12_entity_extraction",
    "stage_13_enhance_clusters",
]
