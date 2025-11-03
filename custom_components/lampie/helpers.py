"""Lampie helpers."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import logging

from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.util import slugify

from .const import CONF_PRIORITY, DOMAIN
from .types import LampieConfigEntry, LampieConfigEntryRuntimeData

_LOGGER = logging.getLogger(__name__)


def is_primary_for_switch(entry: LampieConfigEntry, switch_id: str) -> bool:
    priorities = entry.data.get(CONF_PRIORITY, {}).get(switch_id)

    if not priorities:
        return True
    return bool(priorities[0] == slugify(entry.title))


def get_other_entries(
    config_entry: ConfigEntry, *, hass: HomeAssistant
) -> list[ConfigEntry]:
    return [
        entry
        for entry in hass.config_entries.async_entries(DOMAIN)
        if entry is not config_entry
    ]


@asynccontextmanager
async def auto_reload_disabled(  # noqa: RUF029
    other_entries: list[ConfigEntry],
) -> AsyncGenerator[None]:
    targets: list[LampieConfigEntryRuntimeData] = [
        entry.runtime_data for entry in other_entries if hasattr(entry, "runtime_data")
    ]

    for runtime_data in targets:
        runtime_data.auto_reload_enabled = False

    try:
        yield
    finally:
        for runtime_data in targets:
            runtime_data.auto_reload_enabled = True


@asynccontextmanager
async def unloaded(
    other_entries: list[ConfigEntry],
    *,
    hass: HomeAssistant,
) -> AsyncGenerator[None]:
    other_entries = [  # filter to only the loaded entries
        entry for entry in other_entries if entry.state is ConfigEntryState.LOADED
    ]
    for entry in other_entries:
        await hass.config_entries.async_unload(entry.entry_id)

    yield

    for entry in other_entries:
        assert entry.state is ConfigEntryState.NOT_LOADED
        await hass.config_entries.async_setup(entry.entry_id)
