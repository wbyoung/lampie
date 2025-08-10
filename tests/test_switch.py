"""Test Lampie switches."""

from dataclasses import fields
from typing import Any
from unittest.mock import patch

from homeassistant.components.switch import DOMAIN as SWITCH_DOMAIN
from homeassistant.const import (
    ATTR_ENTITY_ID,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_OFF,
    STATE_ON,
    Platform,
)
from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.restore_state import STORAGE_KEY as RESTORE_STATE_KEY
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_mock_restore_state_shutdown_restart,
    async_mock_service,
    mock_restore_cache_with_extra_data,
    snapshot_platform,
)
from syrupy.assertion import SnapshotAssertion

from custom_components.lampie.switch import LampieSwitch, LampieSwitchDescription
from custom_components.lampie.types import (
    Color,
    Effect,
    LampieNotificationInfo,
    LEDConfig,
)

from . import AbsentNone, setup_integration
from .scenario import Scenario


async def test_switches(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    switch: er.RegistryEntry,
    entity_registry: er.EntityRegistry,
    snapshot: SnapshotAssertion,
) -> None:
    """Test all switches created by the integration."""
    with patch("custom_components.lampie.PLATFORMS", [Platform.SWITCH]):
        await setup_integration(hass, config_entry)

    await snapshot_platform(hass, entity_registry, snapshot, config_entry.entry_id)


async def test_toggle_cycle(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    switch: er.RegistryEntry,
    entity_registry: er.EntityRegistry,
    snapshot: SnapshotAssertion,
) -> None:
    """Test turning on and off a notification."""
    with patch("custom_components.lampie.PLATFORMS", [Platform.SWITCH]):
        await setup_integration(hass, config_entry)

    async_mock_service(hass, "zha", "issue_zigbee_cluster_command")

    await hass.services.async_call(
        SWITCH_DOMAIN,
        SERVICE_TURN_ON,
        {ATTR_ENTITY_ID: "switch.doors_open_notification"},
        blocking=True,
    )

    switch_state = hass.states.get("switch.doors_open_notification")
    assert switch_state.state == STATE_ON

    await hass.services.async_call(
        SWITCH_DOMAIN,
        SERVICE_TURN_OFF,
        {ATTR_ENTITY_ID: "switch.doors_open_notification"},
        blocking=True,
    )

    switch_state = hass.states.get("switch.doors_open_notification")
    assert switch_state.state == STATE_OFF


RESTORE_STATE_SCENARIOS = (
    Scenario(
        "notification_info",
        {
            "storage_entity_id": "switch.doors_open_notification",
            "notification_info": LampieNotificationInfo(
                notification_on=True,
            ),
            "stored_data": {
                "on": True,
                "config_override": None,
            },
            "expected_states": {
                "switch.doors_open_notification": "on",
            },
        },
    ),
    Scenario(
        "notification_info_with_custom_config",
        {
            "storage_entity_id": "switch.doors_open_notification",
            "notification_info": LampieNotificationInfo(
                notification_on=True,
                led_config_override=(LEDConfig(Color.BLUE, effect=Effect.SOLID),),
            ),
            "stored_data": {
                "on": True,
                "config_override": [
                    {
                        "color": "blue",
                        "brightness": 100,
                        "duration": None,
                        "effect": "solid",
                    }
                ],
            },
            "expected_states": {
                "switch.doors_open_notification": "on",
            },
        },
    ),
    Scenario(
        "missing_stored_data",
        {
            "storage_entity_id": "switch.doors_open_notification",
            "notification_info": LampieNotificationInfo(),
            "stored_data": {},
            "expected_stored_data": {
                "on": False,
                "config_override": None,
            },
            "expected_states": {
                "switch.doors_open_notification": "off",
            },
        },
    ),
)


@Scenario.parametrize(*RESTORE_STATE_SCENARIOS)
async def test_restore_sensor_save_state(
    hass: HomeAssistant,
    hass_storage: dict[str, Any],
    config_entry: MockConfigEntry,
    switch: er.RegistryEntry,
    storage_entity_id: str,
    notification_info: LampieNotificationInfo,
    stored_data: dict[str, Any],
    expected_stored_data: dict[str, Any] | AbsentNone,
    expected_states: dict[str, Any],
    snapshot: SnapshotAssertion,
) -> None:
    """Test saving sensor/orchestrator state."""
    await setup_integration(hass, config_entry)

    orchestrator = config_entry.runtime_data.orchestrator
    orchestrator.store_notification_info(
        "doors_open",
        **{
            field.name: getattr(notification_info, field.name)
            for field in fields(notification_info)
        },
    )

    await async_mock_restore_state_shutdown_restart(hass)  # trigger saving state

    stored_entity_data = [
        item["extra_data"]
        for item in hass_storage[RESTORE_STATE_KEY]["data"]
        if item["state"]["entity_id"] == storage_entity_id
    ]

    if expected_stored_data == Scenario.ABSENT:
        expected_stored_data = stored_data

    assert stored_entity_data[0] == expected_stored_data
    assert stored_entity_data == snapshot(name="stored-data")


@Scenario.parametrize(*RESTORE_STATE_SCENARIOS)
async def test_restore_state(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    switch: er.RegistryEntry,
    storage_entity_id: str,
    notification_info: LampieNotificationInfo,
    stored_data: dict[str, Any],
    expected_stored_data: dict[str, Any] | None,
    expected_states: dict[str, Any],
    snapshot: SnapshotAssertion,
) -> None:
    """Test restoring sensor/orchestrator state."""
    mock_restore_cache_with_extra_data(
        hass,
        (
            (
                State(
                    storage_entity_id,
                    "mock-state",  # note: in reality, this would match the stored data
                ),
                stored_data,
            ),
        ),
    )

    await setup_integration(hass, config_entry)

    orchestrator = config_entry.runtime_data.orchestrator
    info = orchestrator.notification_info("doors_open")

    for entity_id, expected_state in expected_states.items():
        assert hass.states.get(entity_id)
        assert hass.states.get(entity_id).state == expected_state

    assert info == snapshot(name="notification_info")


@pytest.mark.usefixtures("init_integration")
def test_restore_functionality_defaults_off(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
):
    description = LampieSwitchDescription(
        key="mock_key",
        value_fn=lambda _: None,
    )
    entity = LampieSwitch(
        description=description,
        coordinator=config_entry.runtime_data.coordinator,
    )

    assert description.is_notification_info_restore_source is False
    assert entity.extra_restore_state_data is None
