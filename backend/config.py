import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    model_name: str
    timeout_seconds: int
    temperature: float

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model_name)


def _to_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _to_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def get_ollama_config() -> OllamaConfig:
    return OllamaConfig(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model_name=os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud"),
        timeout_seconds=_to_int(os.getenv("OLLAMA_TIMEOUT_SECONDS"), 30),
        temperature=_to_float(os.getenv("OLLAMA_TEMPERATURE"), 0.2),
    )
