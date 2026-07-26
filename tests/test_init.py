"""Test component setup."""

from unittest.mock import patch

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    device_registry as dr,
    entity_registry as er,
    issue_registry as ir,
)
from homeassistant.setup import async_setup_component
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
from syrupy.assertion import SnapshotAssertion
from syrupy.filters import props

from custom_components.lampie.const import (
    CONF_COLOR,
    CONF_EFFECT,
    CONF_PRIORITY,
    CONF_SWITCH_ENTITIES,
    DOMAIN,
)

from . import add_mock_switch, setup_integration


async def test_async_setup(hass: HomeAssistant):
    """Test the component gets setup."""
    assert await async_setup_component(hass, DOMAIN, {}) is True


async def test_standard_setup(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    switch: er.RegistryEntry,
    device_registry: dr.DeviceRegistry,
    snapshot: SnapshotAssertion,
) -> None:
    """Test standard setup."""
    await setup_integration(hass, config_entry)

    device = device_registry.async_get(switch.device_id)

    assert device is not None
    assert device.id == switch.device_id
    assert device == snapshot(
        name="device",
        exclude=props(
            # compat for HA DeviceRegistryEntrySnapshot <2025.8.0 and >=2025.6.1
            "suggested_area",
            # compat for HA DeviceRegistryEntrySnapshot <2025.9.0 and >=2025.6.1
            "is_new",
        ),
    )


async def test_config_entries_linked_to_switch_device(
    hass: HomeAssistant,
    device_registry: dr.DeviceRegistry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that the config entries are linked to the switch devices."""
    entryway_switch = add_mock_switch(hass, "light.entryway")
    kitchen_switch = add_mock_switch(hass, "light.kitchen")

    doors_open_entry = MockConfigEntry(
        domain=DOMAIN,
        title="Doors Open",
        entry_id="mock-doors-open-id",
        data={
            CONF_COLOR: "red",
            CONF_EFFECT: "open_close",
            CONF_SWITCH_ENTITIES: [
                "light.entryway",
                "light.kitchen",
                "light.non_existent",
            ],
            CONF_PRIORITY: {
                "light.kitchen": ["medicine", "doors_open"],
                "light.non_existent": ["doors_open", "medicine"],
            },
        },
    )
    doors_open_entry.add_to_hass(hass)
    await setup_integration(hass, doors_open_entry)
    doors_open_device = device_registry.async_get_device(
        {(DOMAIN, doors_open_entry.entry_id)}
    )

    medicine_entry = MockConfigEntry(
        domain=DOMAIN,
        title="Medicine",
        entry_id="mock-medicine-id",
        data={
            CONF_COLOR: "cyan",
            CONF_EFFECT: "slow_blink",
            CONF_SWITCH_ENTITIES: [
                "light.kitchen",
                "light.non_existent",
            ],
            CONF_PRIORITY: {
                "light.kitchen": ["medicine", "doors_open"],
                "light.non_existent": ["doors_open", "medicine"],
            },
        },
    )
    medicine_entry.add_to_hass(hass)
    await setup_integration(hass, medicine_entry)
    medicine_device = device_registry.async_get_device(
        {(DOMAIN, medicine_entry.entry_id)}
    )

    doors_open_device_ids = {
        device.id
        for device in device_registry.devices.values()
        if device.config_entries & {"mock-doors-open-id"}
    }

    medicine_device_ids = {
        device.id
        for device in device_registry.devices.values()
        if device.config_entries & {"mock-medicine-id"}
    }

    assert doors_open_device_ids == {
        doors_open_device.id,
        entryway_switch.device_id,
        kitchen_switch.device_id,
    }

    assert medicine_device_ids == {
        medicine_device.id,
        kitchen_switch.device_id,
    }

    assert (
        "skipping linking of switch to config entry for light.non_existent on "
        "Medicine because an associated device could not be found" in caplog.text
    )

    assert (
        "skipping creation of sensors for light.non_existent on Doors Open "
        "because an associated device could not be found" in caplog.text
    )


async def test_primary_config_entry_sensor_ownership(
    hass: HomeAssistant,
    device_registry: dr.DeviceRegistry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test that sensor is not duplicated across entities on multiple entries."""
    entryway_switch = add_mock_switch(hass, "light.entryway")
    kitchen_switch = add_mock_switch(hass, "light.kitchen")

    doors_open_entry = MockConfigEntry(
        domain=DOMAIN,
        title="Doors Open",
        entry_id="mock-doors-open-id",
        data={
            CONF_COLOR: "red",
            CONF_EFFECT: "open_close",
            CONF_SWITCH_ENTITIES: ["light.entryway", "light.kitchen"],
            CONF_PRIORITY: {
                "light.kitchen": ["medicine", "doors_open"],
            },
        },
    )
    doors_open_entry.add_to_hass(hass)
    await setup_integration(hass, doors_open_entry)
    doors_open_device = device_registry.async_get_device(
        {(DOMAIN, doors_open_entry.entry_id)}
    )

    medicine_entry = MockConfigEntry(
        domain=DOMAIN,
        title="Medicine",
        entry_id="mock-medicine-id",
        data={
            CONF_COLOR: "cyan",
            CONF_EFFECT: "slow_blink",
            CONF_SWITCH_ENTITIES: ["light.kitchen"],
            CONF_PRIORITY: {
                "light.kitchen": ["medicine", "doors_open"],
            },
        },
    )
    medicine_entry.add_to_hass(hass)
    await setup_integration(hass, medicine_entry)
    medicine_device = device_registry.async_get_device(
        {(DOMAIN, medicine_entry.entry_id)}
    )

    doors_open_switch_entities = [
        entity
        for entity in entity_registry.entities.get_entries_for_config_entry_id(
            doors_open_entry.entry_id
        )
        if entity.device_id != doors_open_device.id
    ]
    for entity in doors_open_switch_entities:
        assert entity.device_id == entryway_switch.device_id

    medicine_switch_entities = [
        entity
        for entity in entity_registry.entities.get_entries_for_config_entry_id(
            medicine_entry.entry_id
        )
        if entity.device_id != medicine_device.id
    ]
    for entity in medicine_switch_entities:
        assert entity.device_id == kitchen_switch.device_id

    assert len(doors_open_switch_entities) == len(medicine_switch_entities)


async def test_mismatched_priorities_generate_warning(
    hass: HomeAssistant,
    switch: er.RegistryEntry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that mismatched config priorities generate a warning."""
    doors_open_entry = MockConfigEntry(
        domain=DOMAIN,
        title="Doors Open",
        entry_id="mock-doors-open-id",
        data={
            CONF_COLOR: "red",
            CONF_EFFECT: "open_close",
            CONF_SWITCH_ENTITIES: ["light.kitchen"],
            CONF_PRIORITY: {
                "light.kitchen": ["medicine", "doors_open"],
            },
        },
    )
    doors_open_entry.add_to_hass(hass)
    await setup_integration(hass, doors_open_entry)

    medicine_entry = MockConfigEntry(
        domain=DOMAIN,
        title="Medicine",
        entry_id="mock-medicine-id",
        data={
            CONF_COLOR: "cyan",
            CONF_EFFECT: "slow_blink",
            CONF_SWITCH_ENTITIES: ["light.kitchen"],
            CONF_PRIORITY: {
                "light.kitchen": ["doors_open", "medicine"],
            },
        },
    )
    medicine_entry.add_to_hass(hass)
    await setup_integration(hass, medicine_entry)

    assert (
        "priorities mismatch found for light.kitchen in medicine: "
        "['medicine', 'doors_open'] (previously seen) != "
        "['doors_open', 'medicine'] (for medicine), loading order: 'doors_open'"
    ) in caplog.text


async def test_update_entry(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    await setup_integration(hass, config_entry)

    with patch(
        "homeassistant.config_entries.ConfigEntries.async_reload",
        return_value=True,
    ) as mock_reload:
        hass.config_entries.async_update_entry(
            config_entry,
            data={"arbitrary-update": "value1"},
        )
        await hass.async_block_till_done()
        assert mock_reload.called

        mock_reload.reset_mock()
        config_entry.runtime_data.auto_reload_enabled = False
        hass.config_entries.async_update_entry(
            config_entry,
            data={"arbitrary-update": "value2"},
        )
        await hass.async_block_till_done()
        assert not mock_reload.called


async def test_unload(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    await setup_integration(hass, config_entry)

    assert len(hass.config_entries.async_entries(DOMAIN)) == 1
    assert config_entry.state is ConfigEntryState.LOADED

    assert await hass.config_entries.async_unload(config_entry.entry_id)
    await hass.async_block_till_done()

    assert config_entry.state is ConfigEntryState.NOT_LOADED
    assert not hass.data.get(DOMAIN)


async def test_unload_failure(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> None:
    await setup_integration(hass, config_entry)

    assert len(hass.config_entries.async_entries(DOMAIN)) == 1
    assert config_entry.state is ConfigEntryState.LOADED

    with patch(
        "homeassistant.config_entries.ConfigEntries.async_unload_platforms",
        return_value=False,
    ):
        await hass.config_entries.async_unload(config_entry.entry_id)
        await hass.async_block_till_done()

    assert config_entry.state is ConfigEntryState.FAILED_UNLOAD


async def test_renamed_switch_entity(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    switch: er.RegistryEntry,
    device_registry: dr.DeviceRegistry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test that config entry data is updated when switch entity is renamed."""
    hass.states.async_set(switch.entity_id, "on")

    assert config_entry.data[CONF_SWITCH_ENTITIES] == [switch.entity_id]
    await setup_integration(hass, config_entry)

    entity_registry.async_update_entity(
        switch.entity_id,
        new_entity_id=f"{switch.entity_id}_renamed",
    )
    await hass.async_block_till_done()
    assert config_entry.data[CONF_SWITCH_ENTITIES] == [f"{switch.entity_id}_renamed"]


async def test_device_registry_cleanup(
    hass: HomeAssistant,
    device_registry: dr.DeviceRegistry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that unused linked switch devices are removed from the config entry."""
    entryway_switch = add_mock_switch(hass, "light.entryway")
    kitchen_switch = add_mock_switch(hass, "light.kitchen")

    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Doors Open",
        entry_id="mock-doors-open-id",
        data={
            CONF_COLOR: "red",
            CONF_EFFECT: "open_close",
            CONF_SWITCH_ENTITIES: [entryway_switch.id, kitchen_switch.id],
        },
    )
    entry.add_to_hass(hass)
    await setup_integration(hass, entry)

    service_device = device_registry.async_get_device({(DOMAIN, entry.entry_id)})
    expected_device_ids = {
        service_device.id,
        entryway_switch.device_id,
        kitchen_switch.device_id,
    }

    def get_device_ids() -> set[str]:
        return {
            device.id
            for device in device_registry.devices.values()
            if device.config_entries & {entry.entry_id}
        }

    assert get_device_ids() == expected_device_ids
    assert f"Unlinking device {service_device.id}" not in caplog.text
    assert f"{DOMAIN}.mock-doors-open-id" not in caplog.text
    assert f"Unlinking device {entryway_switch.device_id}" not in caplog.text
    assert f"Unlinking device {kitchen_switch.device_id}" not in caplog.text

    # remove the kitchen switch & check that it was removed from the registry
    hass.config_entries.async_update_entry(
        entry,
        data={**entry.data, CONF_SWITCH_ENTITIES: [entryway_switch.id]},
    )
    expected_device_ids.remove(kitchen_switch.device_id)
    await hass.async_block_till_done()
    assert get_device_ids() == expected_device_ids
    assert f"Unlinking device {service_device.id}" not in caplog.text
    assert f"{DOMAIN}.mock-doors-open-id" not in caplog.text
    assert f"Unlinking device {entryway_switch.device_id}" not in caplog.text
    assert f"Unlinking device {kitchen_switch.device_id}" in caplog.text


async def test_create_removed_switch_entity_issue(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    switch: er.RegistryEntry,
    issue_registry: ir.IssueRegistry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test we create an issue for removed switch entities."""
    hass.states.async_set(switch.entity_id, "not_home")

    await setup_integration(hass, config_entry)

    hass.states.async_remove(switch.entity_id)
    entity_registry.async_remove(switch.entity_id)
    await hass.async_block_till_done()

    assert issue_registry.async_get_issue(
        DOMAIN,
        f"switch_entity_removed_{switch.entity_id}",
    )
