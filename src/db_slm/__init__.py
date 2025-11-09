"""
Database-native Statistical Language Model scaffolding.

This package bootstraps the three-level architecture defined in studies/CONCEPT.md:
    * Level 1 — Aria-backed N-gram lookup tables for statistical generation.
    * Level 2 — Mixed-engine memory tables for episodic context and corrections.
    * Level 3 — Concept prediction plus template-driven verbalisation.

See pipeline.DBSLMEngine for a high-level façade that wires the three levels together.
"""

from .db import DatabaseEnvironment
from .pipeline import DBSLMEngine

__all__ = ["DatabaseEnvironment", "DBSLMEngine"]
