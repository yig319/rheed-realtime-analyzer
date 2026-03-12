from __future__ import annotations

from rheed_core.core.types import ActionRequest
from rheed_core.io.operator_base import Operator


class DummyOperator(Operator):
    """In-memory operator used by tests and local simulations."""

    def __init__(self):
        """Initialize action history storage."""
        self.actions: list[ActionRequest] = []

    def submit(self, action: ActionRequest) -> None:
        """Store submitted action in local history."""
        self.actions.append(action)
