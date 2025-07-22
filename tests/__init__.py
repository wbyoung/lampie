"""Lampie test shared functionality."""

from __future__ import annotations

import datetime as dt
from enum import Flag
from itertools import starmap
from typing import Any

from freezegun.api import FrozenDateTimeFactory
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, entity_registry as er
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
)

ZHA_DOMAIN = "zha"
MOCK_UTC_NOW = dt.datetime(2025, 5, 20, 10, 51, 32, 3245, tzinfo=dt.UTC)


class MockNow:
    def __init__(self, hass: HomeAssistant, freezer: FrozenDateTimeFactory):
        super().__init__()
        self.hass = hass
        self.freezer = freezer

    def _tick(self, seconds) -> None:
        self.freezer.tick(dt.timedelta(seconds=seconds))
        async_fire_time_changed(self.hass)


async def setup_integration(hass: HomeAssistant, config_entry: MockConfigEntry) -> None:
    """Set up the component."""
    config_entry.add_to_hass(hass)
    await setup_added_integration(hass, config_entry)


async def setup_added_integration(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    """Set up a previously added component."""
    await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done()


def add_mock_switch(
    hass, entity_id, device_attrs: dict[str, Any] | None = None
) -> er.RegistryEntry:
    """Add a switch device and (some) related entities.

    Returns:
        The created switch entity.
    """
    domain, object_id = entity_id.split(".")
    device_registry = dr.async_get(hass)
    entity_registry = er.async_get(hass)
    mock_config_entry = MockConfigEntry(
        title=" ".join(object_id.capitalize().split("_")), domain=ZHA_DOMAIN, data={}
    )
    mock_config_entry.add_to_hass(hass)
    device_entry = device_registry.async_get_or_create(
        name=mock_config_entry.title,
        config_entry_id=mock_config_entry.entry_id,
        identifiers={(ZHA_DOMAIN, f"mock-ieee:{object_id}")},
        **(device_attrs or {}),
    )
    switch = entity_registry.async_get_or_create(
        domain,
        ZHA_DOMAIN,
        object_id,
        suggested_object_id=object_id,
        device_id=device_entry.id,
    )
    entity_registry.async_get_or_create(
        "switch",
        ZHA_DOMAIN,
        f"{object_id}-local_protection",
        suggested_object_id=f"{object_id}_local_protection",
        device_id=device_entry.id,
    )
    entity_registry.async_get_or_create(
        "switch",
        ZHA_DOMAIN,
        f"{object_id}-disable_clear_notifications_double_tap",
        suggested_object_id=f"{object_id}_disable_config_2x_tap_to_clear_notifications",
        device_id=device_entry.id,
    )
    return switch


class AbsentNone(Flag):
    """Absent class with singleton for `None` value."""

    _singleton = None


ABSENT_NONE = AbsentNone._singleton


class Scenario:
    """Scenario for repeatable tet parametrization."""

    ABSENT = ABSENT_NONE

    def __init__(self, scenario_id: str, args: dict[str, Any]) -> None:
        super().__init__()
        self._id = scenario_id
        self._args = args

    @classmethod
    def parametrize(cls, *args: Scenario, **kwargs: Any) -> Any:
        scenarios = args + tuple(starmap(Scenario, kwargs.items()))
        ids = []
        argvalues = []
        argnames = tuple(
            {key: None for scenario in scenarios for key in scenario._args}.keys()
        )

        for item in scenarios:
            ids.append(item._id)
            argvalues.append(
                tuple(item._args.get(key, ABSENT_NONE) for key in argnames)
            )

        return pytest.mark.parametrize(argnames=argnames, argvalues=argvalues, ids=ids)
