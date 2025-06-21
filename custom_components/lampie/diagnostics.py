"""Diagnostics support for Lampie."""

from dataclasses import asdict
from typing import Any, cast

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, entity_registry as er

from .const import CONF_SWITCH_ENTITIES
from .coordinator import LampieUpdateCoordinator
from .orchestrator import LampieOrchestrator

TO_REDACT: set[str] = set()


async def async_get_config_entry_diagnostics(  # noqa: RUF029
    hass: HomeAssistant,
    entry: ConfigEntry,
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    coordinator: LampieUpdateCoordinator = entry.runtime_data.coordinator
    orchestrator: LampieOrchestrator = entry.runtime_data.orchestrator
    slug = coordinator.slug
    switches = entry.data[CONF_SWITCH_ENTITIES]
    device_registry = dr.async_get(hass)
    entity_registry = er.async_get(hass)

    return cast(
        "dict[str, Any]",
        async_redact_data(
            {
                "entry": entry.as_dict(),
                "notification": asdict(orchestrator.notification_info(slug)),
                "switches": {
                    switch_id: asdict(orchestrator.switch_info(switch_id))
                    for switch_id in switches
                },
                "switch_registry": {
                    switch_id: entity_registry.async_get(switch_id).as_partial_dict
                    for switch_id in switches
                },
                "device_registry": {
                    switch_id: device_registry.async_get(entity.device_id).dict_repr
                    for switch_id in switches
                    if (entity := entity_registry.async_get(switch_id))
                },
            },
            TO_REDACT,
        ),
    )
