"""
Prompt templates for AutoCognitix AI diagnosis.
"""

from app.prompts.diagnosis_hu import (
    SYSTEM_PROMPT_HU,
    DIAGNOSIS_USER_PROMPT_HU,
    CONFIDENCE_ASSESSMENT_PROMPT_HU,
    RULE_BASED_DIAGNOSIS_PROMPT_HU,
    QUICK_DIAGNOSIS_PROMPT_HU,
    format_dtc_context,
    format_symptom_context,
    format_repair_context,
    format_recall_context,
    build_diagnosis_prompt,
    parse_diagnosis_response,
)

__all__ = [
    "SYSTEM_PROMPT_HU",
    "DIAGNOSIS_USER_PROMPT_HU",
    "CONFIDENCE_ASSESSMENT_PROMPT_HU",
    "RULE_BASED_DIAGNOSIS_PROMPT_HU",
    "QUICK_DIAGNOSIS_PROMPT_HU",
    "format_dtc_context",
    "format_symptom_context",
    "format_repair_context",
    "format_recall_context",
    "build_diagnosis_prompt",
    "parse_diagnosis_response",
]
