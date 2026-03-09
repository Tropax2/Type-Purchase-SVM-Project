"""Top‑level package for custom models.

This file re‑exports the constructors from individual modules so that
users can do ``import models`` and access ``models.svc`` directly.
"""

from .svc import svc  # noqa: F401 - re-exported for convenience

__all__ = ["svc"]
