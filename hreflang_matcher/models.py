from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class LanguageSet:
    """Alle gültigen URLs einer Sprachvariante mit L2-normalisierten Vektoren."""

    code: str                 # hreflang-Code, z. B. "de", "de-AT"
    label: str                # Anzeigename (Dateiname oder Muster)
    urls: list[str]           # Original-URLs
    norm_urls: list[str]      # normalisierte URLs, gleiche Reihenfolge
    vectors: np.ndarray       # Form (n, dim), float32, L2-normalisiert
    dropped: pd.DataFrame     # Spalten: "URL", "Grund"

    def __len__(self) -> int:
        return len(self.urls)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.ndim == 2 else 0


@dataclass
class Match:
    pivot_url: str
    other_url: str
    other_code: str
    score: float              # 1.0 bei Slug-Match, sonst Cosinus
    method: str               # "slug" | "embedding"
    reciprocal: bool
    second_url: str | None    # zweitbester Kandidat derselben Sprache
    second_score: float | None
    margin: float | None      # score - second_score
    confidence: str           # "sicher" | "prüfen"
    reason: str               # leer bei "sicher"
    # Pivot-URL, die die zugeordnete URL selbst am ähnlichsten findet, falls das
    # nicht pivot_url ist (Ursache für "nicht reziprok"); sonst None.
    preferred_pivot_url: str | None = None
    preferred_pivot_score: float | None = None


@dataclass
class MatchResult:
    pivot_code: str
    matches: list[Match]
    unmatched: dict[str, list[str]]        # code -> Original-URLs ohne Zuordnung
    stats: dict[str, dict[str, int]]       # code -> Zähler


@dataclass
class Cluster:
    pivot_url: str
    members: dict[str, str]                # code -> Original-URL, inkl. Pivot
    matches: dict[str, Match]              # code -> Match, ohne Pivot
    confidence: str                        # "sicher" | "prüfen"


@dataclass
class PatternSuggestion:
    pattern: str              # Regex auf die Original-URL
    code: str                 # vorgeschlagener hreflang-Code, evtl. ""
    example: str
    count: int


@dataclass
class EmbeddingConfig:
    """Verbindung zu einem OpenAI-kompatiblen Embedding-Endpunkt."""

    base_url: str              # z. B. "https://api.openai.com/v1", "http://localhost:11434/v1"
    model: str                 # z. B. "text-embedding-3-small", "bge-m3"
    api_key: str = ""          # leer bei lokalen Servern
    max_chars: int = 20000     # Kürzung je Text vor dem Versand
    batch_size: int = 64
    timeout: float = 60.0


@dataclass
class EmbedResult:
    vectors: dict[int, np.ndarray] = field(default_factory=dict)   # Zeilenindex -> float32-Vektor
    errors: dict[int, str] = field(default_factory=dict)           # Zeilenindex -> Grund
    dim: int = 0


@dataclass
class FetchResult:
    texts: dict[int, str] = field(default_factory=dict)            # Zeilenindex -> Haupttext
    errors: dict[int, str] = field(default_factory=dict)           # Zeilenindex -> Grund
