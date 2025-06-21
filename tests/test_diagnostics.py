"""Test Lampie diagnostics."""

from freezegun import freeze_time
from homeassistant.components.switch import DOMAIN as SWITCH_DOMAIN
from homeassistant.const import ATTR_ENTITY_ID, SERVICE_TURN_ON
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_mock_service,
)
from pytest_homeassistant_custom_component.components.diagnostics import (
    get_diagnostics_for_config_entry,
)
from pytest_homeassistant_custom_component.typing import ClientSessionGenerator
from syrupy.assertion import SnapshotAssertion
from syrupy.filters import props

from . import MOCK_UTC_NOW, setup_integration


async def test_entry_diagnostics(
    hass: HomeAssistant,
    hass_client: ClientSessionGenerator,
    config_entry: MockConfigEntry,
    switch: er.RegistryEntry,
    snapshot: SnapshotAssertion,
) -> None:
    """Test config entry diagnostics."""
    with freeze_time(MOCK_UTC_NOW):
        await setup_integration(hass, config_entry)
        async_mock_service(hass, "zha", "issue_zigbee_cluster_command")

        await hass.services.async_call(
            SWITCH_DOMAIN,
            SERVICE_TURN_ON,
            {ATTR_ENTITY_ID: "switch.doors_open_notification"},
            blocking=True,
        )

    assert await get_diagnostics_for_config_entry(
        hass, hass_client, config_entry
    ) == snapshot(
        exclude=props(
            "id",
            "device_id",
            "entry_id",
            "created_at",
            "modified_at",
            "expires_at",
            "config_entries",
            "config_entries_subentries",
            "primary_config_entry",
        )
    )
