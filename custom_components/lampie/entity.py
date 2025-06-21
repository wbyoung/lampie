"""Base support for Lampie entities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
import logging
from typing import Any

from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity import EntityDescription
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import LampieUpdateCoordinator
from .orchestrator import LampieOrchestrator

_LOGGER = logging.getLogger(__name__)


class LampieEntity[DataT](ABC, CoordinatorEntity[LampieUpdateCoordinator]):
    """Lampie entity class."""

    _attr_has_entity_name = True
    entity_description: LampieEntityDescription

    def __init__(
        self,
        *,
        description: LampieEntityDescription,
        coordinator: LampieUpdateCoordinator,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)

        self.entity_description = description

        self._attr_unique_id = f"{coordinator.config_entry.entry_id}_{description.key}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, coordinator.config_entry.entry_id)},
            name=coordinator.config_entry.title,
            entry_type=DeviceEntryType.SERVICE,
        )

    @property
    @abstractmethod
    def data(self) -> DataT:
        """Get data from coordinator."""

    @property
    def orchestrator(self) -> LampieOrchestrator:
        """Get the Lapmpie orchestrator."""
        config_entry = self.coordinator.config_entry
        orchestrator: LampieOrchestrator = config_entry.runtime_data.orchestrator
        return orchestrator

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return super().available and self.data is not None

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        """Return the state attributes."""
        return self.entity_description.attrs_fn(self.data)


class LampieDistributedEntity[DataT](LampieEntity[DataT]):
    """Lampie entity class for those distributed across several switches."""

    _attr_has_entity_name = True
    entity_description: LampieEntityDescription

    def __init__(
        self,
        *,
        description: LampieEntityDescription,
        coordinator: LampieUpdateCoordinator,
        switch_id: str,
        switch_device_info: DeviceInfo,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(description=description, coordinator=coordinator)

        self.switch_id = switch_id

        id_tuple = next(iter(switch_device_info["identifiers"]))
        id_slug = "_".join(id_tuple)

        self._attr_unique_id = f"{id_slug}_{description.key}"
        self._attr_device_info = switch_device_info


class LampieEntityDescription[DataT](EntityDescription):
    """Class describing Lampie entities."""

    value_fn: Callable[
        [DataT],
        str | int | float | None,
    ]
    attrs_fn: Callable[
        [DataT],
        dict[str, Any] | None,
    ] = lambda _: None
