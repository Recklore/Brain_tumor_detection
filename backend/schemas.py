from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RiskLevel(str, Enum):
    low = "low"
    moderate = "moderate"
    high = "high"


class StructuredReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    diagnosis_summary: str = Field(..., description="One-line summary of likely diagnosis")
    risk_level: RiskLevel = Field(..., description="Overall risk estimate")
    key_findings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    confidence_note: str = Field(..., description="Short confidence interpretation")
    disclaimer: str = Field(
        default="AI-generated summary for informational use only. Not a clinical diagnosis.",
        description="Safety disclaimer",
    )

    @field_validator("diagnosis_summary", "confidence_note", "disclaimer", mode="before")
    @classmethod
    def _clean_text(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("key_findings", "recommendations", mode="before")
    @classmethod
    def _clean_list(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            return []

        cleaned: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                cleaned.append(text)
        return cleaned[:4]
