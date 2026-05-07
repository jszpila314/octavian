from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _merge_selected_columns(
    resolved: dict[str, Any],
    conditional_columns: Mapping[str, Any] | None,
    selector: str | None,
) -> None:
    if not conditional_columns or selector is None:
        return

    selected_columns = conditional_columns.get(selector)
    if isinstance(selected_columns, Mapping):
        resolved.update(selected_columns)


def resolve_dataset_columns(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return output dataset mappings enabled for this run configuration."""
    dataset_columns = config.get("dataset_columns", {})
    resolved = dict(dataset_columns) if isinstance(dataset_columns, Mapping) else {}

    _merge_selected_columns(
        resolved,
        config.get("dataset_columns_by_halo_source"),
        config.get("halo_source"),
    )
    _merge_selected_columns(
        resolved,
        config.get("dataset_columns_by_halo_mode"),
        config.get("halo_mode"),
    )

    return resolved


def resolve_list_dataset_columns(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return variable-length output dataset mappings enabled for this run."""
    dataset_columns = config.get("list_dataset_columns", {})
    resolved = dict(dataset_columns) if isinstance(dataset_columns, Mapping) else {}

    _merge_selected_columns(
        resolved,
        config.get("list_dataset_columns_by_halo_source"),
        config.get("halo_source"),
    )
    _merge_selected_columns(
        resolved,
        config.get("list_dataset_columns_by_halo_mode"),
        config.get("halo_mode"),
    )

    return resolved
