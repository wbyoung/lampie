"""The Lampie integration."""

from __future__ import annotations

import logging

from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.event import async_track_entity_registry_updated_event
from homeassistant.helpers.typing import ConfigType

from .const import CONF_SWITCH_ENTITIES, DOMAIN
from .coordinator import LampieUpdateCoordinator
from .orchestrator import LampieOrchestrator
from .services import async_setup_services
from .types import LampieConfigEntry, LampieConfigEntryRuntimeData

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.SENSOR, Platform.SWITCH]
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


async def async_setup(  # noqa: RUF029
    hass: HomeAssistant,
    config: ConfigType,  # noqa: ARG001
) -> bool:
    """Set up Lampie services.

    Returns:
        If the setup was successful.
    """
    async_setup_services(hass)

    return True


async def async_setup_entry(hass: HomeAssistant, entry: LampieConfigEntry) -> bool:
    """Set up Lampie from a config entry.

    Returns:
        If the setup was successful.
    """
    _LOGGER.debug("setup %s with config:%s", entry.title, entry.data)

    if DOMAIN not in hass.data:
        orchestrator = LampieOrchestrator(hass)
        hass.data[DOMAIN] = orchestrator

        await orchestrator.setup()

    coordinator = LampieUpdateCoordinator(hass, entry)
    orchestrator = hass.data[DOMAIN]
    orchestrator.add_coordinator(coordinator)
    entry.runtime_data = LampieConfigEntryRuntimeData(
        orchestrator=orchestrator,
        coordinator=coordinator,
    )

    entry.async_on_unload(
        async_track_entity_registry_updated_event(
            hass,
            entry.data[CONF_SWITCH_ENTITIES],
            coordinator.async_handle_switch_id_change,
        ),
    )

    await coordinator.async_config_entry_first_refresh()
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: LampieConfigEntry) -> bool:
    """Unload a config entry.

    Returns:
        If the unload was successful.
    """
    coordinator: LampieUpdateCoordinator = entry.runtime_data.coordinator
    unload_ok: bool = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if orchestrator := entry.runtime_data.orchestrator:
        orchestrator.remove_coordinator(coordinator)

        if orchestrator.teardown() and orchestrator == hass.data.get(DOMAIN):
            hass.data.pop(DOMAIN)

    return unload_ok


async def _async_update_listener(
    hass: HomeAssistant,
    entry: LampieConfigEntry,
) -> None:
    """Handle options update."""
    runtime_data: LampieConfigEntryRuntimeData = entry.runtime_data

    if runtime_data.auto_reload_enabled:
        await hass.config_entries.async_reload(entry.entry_id)
