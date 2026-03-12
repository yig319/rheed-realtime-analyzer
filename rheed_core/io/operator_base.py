from __future__ import annotations

"""Operator interfaces for submitting advisory/command requests."""

from abc import ABC, abstractmethod

from rheed_core.core.types import ActionRequest


class Operator(ABC):
    """Abstract sink for policy-generated actions."""

    @abstractmethod
    def submit(self, action: ActionRequest) -> None:
        """Handle one action request."""
        raise NotImplementedError
