"""Compatibility path for the released pooled-feature checkpoints."""

from .model import LegacyPooledConfig, LegacyPooledEpiCataNexus, load_legacy_checkpoint

__all__ = ["LegacyPooledConfig", "LegacyPooledEpiCataNexus", "load_legacy_checkpoint"]
