"""Test Lampie sensors."""

from dataclasses import fields
from typing import Any
from unittest.mock import patch

from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.restore_state import STORAGE_KEY as RESTORE_STATE_KEY
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_mock_restore_state_shutdown_restart,
    mock_restore_cache_with_extra_data,
    snapshot_platform,
)
from syrupy.assertion import SnapshotAssertion

from custom_components.lampie.sensor import LampieSensor, LampieSensorDescription
from custom_components.lampie.types import (
    Color,
    Effect,
    LampieSwitchInfo,
    LEDConfig,
    LEDConfigSource,
    LEDConfigSourceType,
)

from . import AbsentNone, setup_integration
from .scenario import Scenario


async def test_sensors(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    switch: er.RegistryEntry,
    entity_registry: er.EntityRegistry,
    snapshot: SnapshotAssertion,
) -> None:
    """Test all sensors created by the integration."""
    with patch("custom_components.lampie.PLATFORMS", [Platform.SENSOR]):
        await setup_integration(hass, config_entry)

    await snapshot_platform(hass, entity_registry, snapshot, config_entry.entry_id)


async def test_sensors_with_simple_led_config(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    switch: er.RegistryEntry,
    entity_registry: er.EntityRegistry,
    snapshot: SnapshotAssertion,
) -> None:
    """Test all sensors created by the integration with simple LED config."""
    with patch("custom_components.lampie.PLATFORMS", [Platform.SENSOR]):
        await setup_integration(hass, config_entry)

    config_entry.runtime_data.orchestrator.store_switch_info(
        switch.entity_id,
        led_config_source=LEDConfigSource("medicine"),
        led_config=(
            LEDConfig(
                color=Color.CYAN,
                effect=Effect.SLOW_BLINK,
            ),
        ),
    )

    await snapshot_platform(hass, entity_registry, snapshot, config_entry.entry_id)


async def test_sensors_with_advanced_led_config(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    switch: er.RegistryEntry,
    entity_registry: er.EntityRegistry,
    snapshot: SnapshotAssertion,
) -> None:
    """Test all sensors created by the integration with advanced LED config."""
    with patch("custom_components.lampie.PLATFORMS", [Platform.SENSOR]):
        await setup_integration(hass, config_entry)

    config_entry.runtime_data.orchestrator.store_switch_info(
        switch.entity_id,
        led_config_source=LEDConfigSource("doors_open"),
        led_config=(
            LEDConfig(
                color=Color.RED,
                brightness=80.0,
                effect=Effect.SLOW_BLINK,
                duration=5,
            ),
            LEDConfig(
                color=Color.ORANGE,
                brightness=20.0,
                effect=Effect.SOLID,
                duration=10,
            ),
            LEDConfig(
                color=Color.YELLOW,
                brightness=100.0,
                effect=Effect.SLOW_BLINK,
                duration=5,
            ),
            LEDConfig(
                color=Color.RED + 1,
                brightness=80.0,
                effect=Effect.SLOW_BLINK,
                duration=5,
            ),
            LEDConfig(
                color=Color.ORANGE - 1,
                brightness=20.0,
                effect=Effect.SOLID,
                duration=10,
            ),
            LEDConfig(
                color=int(Color.YELLOW),
                brightness=80.0,
                effect=Effect.SLOW_BLINK,
                duration=5,
            ),
            LEDConfig(
                color=Color.WHITE,
                brightness=100.0,
                effect=Effect.SOLID,
                duration=10,
            ),
        ),
    )

    await snapshot_platform(hass, entity_registry, snapshot, config_entry.entry_id)


async def test_missing_switch_device(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    device_registry: dr.DeviceRegistry,
    entity_registry: er.EntityRegistry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that the config entries are linked to the switch devices."""
    with patch("custom_components.lampie.PLATFORMS", [Platform.SENSOR]):
        await setup_integration(hass, config_entry)

    devices = [
        device
        for device in device_registry.devices.values()
        if config_entry.entry_id in device.config_entries
    ]

    entities = entity_registry.entities.get_entries_for_config_entry_id(
        config_entry.entry_id
    )

    assert len(devices) == 0
    assert len(entities) == 0
    assert "skipping creation of sensors for light.kitchen on Doors Open" in caplog.text


RESTORE_STATE_SCENARIOS = (
    Scenario(
        "switch_info",
        {
            "storage_entity_id": "sensor.kitchen_notification",
            "switch_info": LampieSwitchInfo(
                led_config=(
                    LEDConfig(
                        Color.RED, effect=Effect.SOLID, duration=4, brightness=50.0
                    ),
                ),
                led_config_source=LEDConfigSource("doors_open"),
                local_protection_id="unstored:entity_id",
                disable_clear_notification_id="unstored:entity_id",
                priorities=("doors_open",),
            ),
            "stored_data": {
                "config": [
                    {
                        "color": "red",
                        "brightness": 50.0,
                        "duration": 4,
                        "effect": "solid",
                    }
                ],
                "source": {
                    "type": "notification",
                    "value": "doors_open",
                },
            },
            "expected_states": {
                "sensor.kitchen_notification": "doors_open",
                "sensor.kitchen_effect_color": "red",
                "sensor.kitchen_effect_type": "solid",
                "sensor.kitchen_effect_duration": "4",
                "sensor.kitchen_effect_brightness": "50.0",
            },
        },
    ),
    Scenario(
        "switch_info_with_numeric_color",
        {
            "storage_entity_id": "sensor.kitchen_notification",
            "switch_info": LampieSwitchInfo(
                led_config=(
                    LEDConfig(80, effect=Effect.SOLID, duration=4, brightness=50.0),
                ),
                led_config_source=LEDConfigSource("doors_open"),
                local_protection_id="unstored:entity_id",
                disable_clear_notification_id="unstored:entity_id",
                priorities=("doors_open",),
            ),
            "stored_data": {
                "config": [
                    {
                        "color": 80,
                        "brightness": 50.0,
                        "duration": 4,
                        "effect": "solid",
                    }
                ],
                "source": {
                    "type": "notification",
                    "value": "doors_open",
                },
            },
            "expected_states": {
                "sensor.kitchen_notification": "doors_open",
                "sensor.kitchen_effect_color": "80",
                "sensor.kitchen_effect_type": "solid",
                "sensor.kitchen_effect_duration": "4",
                "sensor.kitchen_effect_brightness": "50.0",
            },
        },
    ),
    Scenario(
        "switch_info_for_override_service_call",
        {
            "storage_entity_id": "sensor.kitchen_notification",
            "switch_info": LampieSwitchInfo(
                led_config=(LEDConfig(Color.BLUE, effect=Effect.SOLID),),
                led_config_source=LEDConfigSource(
                    "lampie.override", LEDConfigSourceType.SERVICE
                ),
                local_protection_id="unstored:entity_id",
                disable_clear_notification_id="unstored:entity_id",
                priorities=("doors_open",),
            ),
            "stored_data": {
                "config": [
                    {
                        "color": "blue",
                        "brightness": 100.0,
                        "duration": None,
                        "effect": "solid",
                    }
                ],
                "source": {
                    "type": "service",
                    "value": "lampie.override",
                },
            },
            "expected_states": {
                "sensor.kitchen_notification": "lampie.override",
                "sensor.kitchen_effect_color": "blue",
                "sensor.kitchen_effect_type": "solid",
                "sensor.kitchen_effect_duration": "unknown",
                "sensor.kitchen_effect_brightness": "100.0",
            },
        },
    ),
    Scenario(
        "missing_stored_data",
        {
            "storage_entity_id": "sensor.kitchen_notification",
            "switch_info": LampieSwitchInfo(
                led_config=(),
                led_config_source=LEDConfigSource(None),
                priorities=("doors_open",),
            ),
            "stored_data": {},
            "expected_stored_data": {
                "config": [],
                "source": {
                    "type": "notification",
                    "value": None,
                },
            },
            "expected_states": {
                "sensor.kitchen_notification": "unknown",
                "sensor.kitchen_effect_color": "unknown",
                "sensor.kitchen_effect_type": "unknown",
                "sensor.kitchen_effect_duration": "unknown",
                "sensor.kitchen_effect_brightness": "unknown",
            },
        },
    ),
    *[
        scenario
        for storage_entity_id in [
            "sensor.kitchen_effect_color",
            "sensor.kitchen_effect_type",
            "sensor.kitchen_effect_duration",
            "sensor.kitchen_effect_brightness",
        ]
        for scenario in [
            Scenario(
                f"no_restore_for_{storage_entity_id}",
                {
                    "storage_entity_id": storage_entity_id,
                    "switch_info": LampieSwitchInfo(
                        led_config=(LEDConfig(Color.BLUE, effect=Effect.SOLID),),
                        led_config_source=LEDConfigSource("doors_open"),
                        local_protection_id="unstored:entity_id",
                        disable_clear_notification_id="unstored:entity_id",
                        priorities=("doors_open",),
                    ),
                    "stored_data": {
                        "config": [
                            {
                                "color": "blue",
                                "brightness": 100.0,
                                "duration": None,
                                "effect": "solid",
                            }
                        ],
                        "source": {
                            "type": "notification",
                            "value": "doors_open",
                        },
                    },
                    "expected_stored_data": None,
                    "expected_states": {
                        "sensor.kitchen_notification": "unknown",
                        "sensor.kitchen_effect_color": "unknown",
                        "sensor.kitchen_effect_type": "unknown",
                        "sensor.kitchen_effect_duration": "unknown",
                        "sensor.kitchen_effect_brightness": "unknown",
                    },
                },
            ),
        ]
    ],
)


@Scenario.parametrize(*RESTORE_STATE_SCENARIOS)
async def test_restore_sensor_save_state(
    hass: HomeAssistant,
    hass_storage: dict[str, Any],
    config_entry: MockConfigEntry,
    switch: er.RegistryEntry,
    storage_entity_id: str,
    switch_info: LampieSwitchInfo,
    stored_data: dict[str, Any],
    expected_stored_data: dict[str, Any] | AbsentNone,
    expected_states: dict[str, Any],
    snapshot: SnapshotAssertion,
) -> None:
    """Test saving sensor/orchestrator state."""
    await setup_integration(hass, config_entry)

    orchestrator = config_entry.runtime_data.orchestrator
    orchestrator.store_switch_info(
        switch.entity_id,
        **{
            field.name: getattr(switch_info, field.name)
            for field in fields(switch_info)
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
    switch_info: LampieSwitchInfo,
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
    info = orchestrator.switch_info(switch.entity_id)

    for entity_id, expected_state in expected_states.items():
        assert hass.states.get(entity_id)
        assert hass.states.get(entity_id).state == expected_state

    assert info == snapshot(name="switch_info")


@pytest.mark.usefixtures("init_integration")
def test_restore_functionality_defaults_off(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
):
    description = LampieSensorDescription(
        key="mock_key",
        value_fn=lambda _: None,
    )
    entity = LampieSensor(
        description=description,
        coordinator=config_entry.runtime_data.coordinator,
        switch_id="mock_switch_id",
        switch_device_info=DeviceInfo(
            identifiers={("mock-device-domain", "mock-device-id")},
        ),
    )

    assert description.is_switch_info_restore_source is False
    assert entity.extra_restore_state_data is None
