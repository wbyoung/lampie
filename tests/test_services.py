"""Test services for the Lampie integration."""

import re
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import entity_registry as er
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_mock_service,
)
from syrupy.assertion import SnapshotAssertion
import voluptuous as vol

from custom_components.lampie.const import (
    ATTR_ENTITY_ID,
    ATTR_LED_CONFIG,
    ATTR_NOTIFICATION,
    DOMAIN,
)
from custom_components.lampie.services import (
    SERVICE_NAME_ACTIVATE,
    SERVICE_NAME_OVERRIDE,
)


@pytest.mark.usefixtures("init_integration")
def test_has_services(
    hass: HomeAssistant,
) -> None:
    """Test the existence of the Lampie Service."""
    assert hass.services.has_service(DOMAIN, SERVICE_NAME_ACTIVATE)
    assert hass.services.has_service(DOMAIN, SERVICE_NAME_OVERRIDE)


@pytest.mark.usefixtures("switch", "init_integration")
@pytest.mark.parametrize(
    "attrs",
    [
        {ATTR_NOTIFICATION: "doors_open"},
        {ATTR_NOTIFICATION: "doors_open", ATTR_LED_CONFIG: "green"},
        {
            ATTR_NOTIFICATION: "doors_open",
            ATTR_LED_CONFIG: [
                "green",
                "orange",
                "blue",
                "yellow",
                "pink",
                "red",
                {
                    "color": "cyan",
                    "effect": "fast_blink",
                },
            ],
        },
    ],
    ids=["notification_only", "led_color", "full_led_config"],
)
async def test_activate_service(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    switch: er.RegistryEntry,
    attrs: dict[str, Any],
    entity_registry: er.EntityRegistry,
    snapshot: SnapshotAssertion,
):
    """Test `activate` service call."""
    cluster_commands = async_mock_service(hass, "zha", "issue_zigbee_cluster_command")

    await hass.services.async_call(
        DOMAIN,
        SERVICE_NAME_ACTIVATE,
        attrs,
        blocking=True,
    )

    assert cluster_commands == snapshot(name="zha_cluster_commands")

    for entity in entity_registry.entities.get_entries_for_config_entry_id(
        config_entry.entry_id
    ):
        assert hass.states.get(entity.entity_id) == snapshot(name=entity.entity_id)


@pytest.mark.usefixtures("init_integration")
@pytest.mark.parametrize(
    "attrs",
    [
        {ATTR_ENTITY_ID: "light.kitchen", ATTR_LED_CONFIG: "green"},
        {
            ATTR_ENTITY_ID: "light.kitchen",
            ATTR_LED_CONFIG: [
                "green",
                "orange",
                "blue",
                "yellow",
                "pink",
                "red",
                {
                    "color": "cyan",
                    "effect": "fast_blink",
                },
            ],
        },
    ],
    ids=["led_color", "full_led_config"],
)
async def test_override_service(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    switch: er.RegistryEntry,
    attrs: dict[str, Any],
    entity_registry: er.EntityRegistry,
    snapshot: SnapshotAssertion,
):
    """Test `override` service call."""
    cluster_commands = async_mock_service(hass, "zha", "issue_zigbee_cluster_command")

    await hass.services.async_call(
        DOMAIN,
        SERVICE_NAME_OVERRIDE,
        attrs,
        blocking=True,
    )

    assert cluster_commands == snapshot(name="zha_cluster_commands")

    for entity in entity_registry.entities.get_entries_for_config_entry_id(
        config_entry.entry_id
    ):
        assert hass.states.get(entity.entity_id) == snapshot(name=entity.entity_id)


@pytest.mark.usefixtures("init_integration")
@pytest.mark.parametrize(
    ("service_data", "error", "error_message"),
    [
        ({}, vol.er.Error, "required key not provided .+"),
        (
            {"notification": "doors_open", "leds": "pinkish"},
            vol.er.Error,
            ".+ not a valid color.+name",
        ),
        (
            {"notification": "incorrect_slug"},
            ServiceValidationError,
            "Invalid notification .+",
        ),
    ],
)
async def test_activate_service_validation(
    hass: HomeAssistant,
    service_data: dict[str, str],
    error: type[Exception],
    error_message: str,
) -> None:
    """Test the `activate` service validation."""
    with pytest.raises(error) as exc:
        await hass.services.async_call(
            DOMAIN,
            SERVICE_NAME_ACTIVATE,
            service_data,
            blocking=True,
        )
    assert re.match(error_message, str(exc.value))


@pytest.mark.usefixtures("init_integration")
@pytest.mark.parametrize(
    ("service_data", "error", "error_message"),
    [
        ({"entity_id": "light.kitchen"}, vol.er.Error, "required key not provided .+"),
        ({"leds": "green"}, vol.er.Error, "required key not provided .+"),
        (
            {"entity_id": "light.kitchen", "leds": "pinkish"},
            vol.er.Error,
            ".+ not a valid color.+name",
        ),
    ],
)
async def test_override_service_validation(
    hass: HomeAssistant,
    service_data: dict[str, str],
    error: type[Exception],
    error_message: str,
) -> None:
    """Test the `override` service validation."""
    with pytest.raises(error) as exc:
        await hass.services.async_call(
            DOMAIN,
            SERVICE_NAME_OVERRIDE,
            service_data,
            blocking=True,
        )
    assert re.match(error_message, str(exc.value))
