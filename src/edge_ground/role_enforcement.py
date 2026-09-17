"""
Deployment Role Enforcement Guards — Module 21.

Provides strict static and runtime deployment-role guards.
Enforces component ownership:
- EDGE role cannot initialize ML models, RUL, Advisory, What-If, or Ground REST API services.
- GROUND role cannot initialize Edge acquisition nodes.
- SIMULATION role permits full local dev/testing end-to-end execution.
"""

from __future__ import annotations

from typing import Callable, Any
from functools import wraps

from src.core.config import AppSettings, get_settings
from src.core.exceptions import RoleViolationError
from src.core.provenance import DeploymentRole


def check_role_permission(
    allowed_roles: list[DeploymentRole],
    component_name: str,
    settings: AppSettings | None = None,
) -> None:
    """Verify that current deployment role is permitted for component_name."""
    current_settings = settings or get_settings()
    current_role = current_settings.deployment.role

    # SIMULATION role allows all component execution for dev/test
    if current_role == DeploymentRole.SIMULATION:
        return

    if current_role not in allowed_roles:
        raise RoleViolationError(
            f"Component '{component_name}' is restricted to roles {[r.value for r in allowed_roles]}, "
            f"but current node is configured with DeploymentRole.{current_role.value}."
        )


def require_deployment_role(*allowed_roles: DeploymentRole) -> Callable:
    """Decorator guarding functions to execute only under specific deployment roles."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            check_role_permission(list(allowed_roles), func.__qualname__)
            return func(*args, **kwargs)
        return wrapper
    return decorator
