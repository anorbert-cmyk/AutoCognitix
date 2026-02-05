"""
Prompt templates for AutoCognitix AI diagnosis.
"""

from app.prompts.diagnosis_hu import (
    CONFIDENCE_ASSESSMENT_PROMPT_HU,
    DIAGNOSIS_USER_PROMPT_HU,
    QUICK_DIAGNOSIS_PROMPT_HU,
    RULE_BASED_DIAGNOSIS_PROMPT_HU,
    SYSTEM_PROMPT_HU,
    build_diagnosis_prompt,
    format_dtc_context,
    format_recall_context,
    format_repair_context,
    format_symptom_context,
    parse_diagnosis_response,
)

__all__ = [
    "CONFIDENCE_ASSESSMENT_PROMPT_HU",
    "DIAGNOSIS_USER_PROMPT_HU",
    "QUICK_DIAGNOSIS_PROMPT_HU",
    "RULE_BASED_DIAGNOSIS_PROMPT_HU",
    "SYSTEM_PROMPT_HU",
    "build_diagnosis_prompt",
    "format_dtc_context",
    "format_recall_context",
    "format_repair_context",
    "format_symptom_context",
    "parse_diagnosis_response",
]
