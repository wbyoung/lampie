"""Test Lampie config flow."""

from contextlib import contextmanager
from unittest.mock import patch

from homeassistant.config_entries import (
    SOURCE_USER,
    ConfigEntry,
    ConfigEntryDisabler,
    ConfigEntryState,
)
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import device_registry as dr, entity_registry as er, selector
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.lampie.config_flow import SECTION_ADVANCED_OPTIONS
from custom_components.lampie.const import (
    CONF_COLOR,
    CONF_DURATION,
    CONF_EFFECT,
    CONF_LED_CONFIG,
    CONF_NAME,
    CONF_PRIORITY,
    CONF_SWITCH_ENTITIES,
    DOMAIN,
)
from custom_components.lampie.types import LampieConfigEntryRuntimeData

from . import Scenario, add_mock_switch, setup_added_integration

FLOW_SCENARIOS = {
    "non_overlapping": {
        "user_input": {
            CONF_NAME: "Medicine",
            CONF_COLOR: "cyan",
            CONF_EFFECT: "slow_blink",
            CONF_SWITCH_ENTITIES: ["light.dining_room"],
            CONF_DURATION: "4:00",
        },
        "priority_inputs": [],
        "expected_result": {
            "data": {
                CONF_NAME: "Medicine",
                CONF_COLOR: "cyan",
                CONF_EFFECT: "slow_blink",
                CONF_SWITCH_ENTITIES: ["light.dining_room"],
                CONF_DURATION: "4:00",
            }
        },
    },
    "overlap_on_single_light": {
        "user_input": {
            CONF_NAME: "Medicine",
            CONF_COLOR: "cyan",
            CONF_EFFECT: "slow_blink",
            CONF_SWITCH_ENTITIES: ["light.kitchen"],
        },
        "priority_inputs": [
            [
                "doors_open",
                "medicine",
            ]
        ],
        "expected_result": {
            "data": {
                CONF_NAME: "Medicine",
                CONF_COLOR: "cyan",
                CONF_EFFECT: "slow_blink",
                CONF_SWITCH_ENTITIES: ["light.kitchen"],
                CONF_PRIORITY: {
                    "light.kitchen": [
                        "doors_open",
                        "medicine",
                    ]
                },
            }
        },
    },
    "valid_color_number": {
        "user_input": {
            CONF_NAME: "Medicine",
            CONF_COLOR: "25",
            CONF_EFFECT: "slow_blink",
            CONF_SWITCH_ENTITIES: ["light.dining_room"],
            CONF_DURATION: "4:00",
        },
        "priority_inputs": [],
        "expected_result": {
            "data": {
                CONF_NAME: "Medicine",
                CONF_COLOR: 25,
                CONF_EFFECT: "slow_blink",
                CONF_SWITCH_ENTITIES: ["light.dining_room"],
                CONF_DURATION: "4:00",
            }
        },
    },
    "valid_led_config": {
        "user_input": {
            CONF_NAME: "Medicine",
            CONF_COLOR: "",
            CONF_SWITCH_ENTITIES: ["light.dining_room"],
            SECTION_ADVANCED_OPTIONS: {
                CONF_LED_CONFIG: [
                    {
                        CONF_COLOR: "red",
                        CONF_EFFECT: "slow_blink",
                        CONF_DURATION: "4:00",
                    },
                    {
                        CONF_COLOR: "orange",
                        CONF_EFFECT: "slow_blink",
                        CONF_DURATION: "4:00",
                    },
                    {
                        CONF_COLOR: "white",
                        CONF_EFFECT: "slow_blink",
                        CONF_DURATION: "4:00",
                    },
                    {
                        CONF_COLOR: "red",
                        CONF_EFFECT: "slow_blink",
                        CONF_DURATION: "4:00",
                    },
                    {
                        CONF_COLOR: "orange",
                        CONF_EFFECT: "slow_blink",
                        CONF_DURATION: "4:00",
                    },
                    {
                        CONF_COLOR: "white",
                        CONF_EFFECT: "slow_blink",
                        CONF_DURATION: "4:00",
                    },
                    {CONF_COLOR: 0, CONF_EFFECT: "slow_blink", CONF_DURATION: "4:00"},
                ]
            },
        },
        "priority_inputs": [],
        "expected_result": {
            "data": {
                CONF_NAME: "Medicine",
                CONF_SWITCH_ENTITIES: ["light.dining_room"],
                CONF_LED_CONFIG: [
                    {
                        CONF_COLOR: "red",
                        CONF_EFFECT: "slow_blink",
                        CONF_DURATION: "4:00",
                    },
                    {
                        CONF_COLOR: "orange",
                        CONF_EFFECT: "slow_blink",
                        CONF_DURATION: "4:00",
                    },
                    {
                        CONF_COLOR: "white",
                        CONF_EFFECT: "slow_blink",
                        CONF_DURATION: "4:00",
                    },
                    {
                        CONF_COLOR: "red",
                        CONF_EFFECT: "slow_blink",
                        CONF_DURATION: "4:00",
                    },
                    {
                        CONF_COLOR: "orange",
                        CONF_EFFECT: "slow_blink",
                        CONF_DURATION: "4:00",
                    },
                    {
                        CONF_COLOR: "white",
                        CONF_EFFECT: "slow_blink",
                        CONF_DURATION: "4:00",
                    },
                    {CONF_COLOR: 0, CONF_EFFECT: "slow_blink", CONF_DURATION: "4:00"},
                ],
            }
        },
    },
    "valid_led_config_abbreviated": {
        "user_input": {
            CONF_NAME: "Medicine",
            CONF_COLOR: "",
            CONF_SWITCH_ENTITIES: ["light.dining_room"],
            SECTION_ADVANCED_OPTIONS: {
                CONF_LED_CONFIG: [
                    "red",
                    "orange",
                    "white",
                    250,
                    20,
                    255,
                    0,
                ]
            },
        },
        "priority_inputs": [],
        "expected_result": {
            "data": {
                CONF_NAME: "Medicine",
                CONF_SWITCH_ENTITIES: ["light.dining_room"],
                CONF_LED_CONFIG: [
                    "red",
                    "orange",
                    "white",
                    250,
                    20,
                    255,
                    0,
                ],
            }
        },
    },
    "missing_color": {
        "user_input": {
            CONF_NAME: "Medicine",
            CONF_COLOR: "",
            CONF_EFFECT: "slow_blink",
            CONF_SWITCH_ENTITIES: ["light.dining_room"],
            CONF_DURATION: "4:00",
        },
        "priority_inputs": [],
        "expected_result": {
            "errors": {CONF_COLOR: "missing_color"},
            "description_placeholders": {
                "config_title": "Medicine",
            },
        },
    },
    "missing_effect": {
        "user_input": {
            CONF_NAME: "Medicine",
            CONF_COLOR: "cyan",
            CONF_SWITCH_ENTITIES: ["light.dining_room"],
            CONF_DURATION: "4:00",
        },
        "priority_inputs": [],
        "expected_result": {
            "errors": {CONF_EFFECT: "missing_effect"},
            "description_placeholders": {
                "config_title": "Medicine",
            },
        },
    },
    "invalid_color_string": {
        "user_input": {
            CONF_NAME: "Medicine",
            CONF_COLOR: "pinkish",
            CONF_EFFECT: "slow_blink",
            CONF_SWITCH_ENTITIES: ["light.dining_room"],
            CONF_DURATION: "4:00",
        },
        "priority_inputs": [],
        "expected_result": {
            "errors": {CONF_COLOR: "invalid_color_name"},
            "description_placeholders": {
                "config_title": "Medicine",
                "color": "pinkish",
            },
        },
    },
    "invalid_color_number": {
        "user_input": {
            CONF_NAME: "Medicine",
            CONF_COLOR: "521",
            CONF_EFFECT: "slow_blink",
            CONF_SWITCH_ENTITIES: ["light.dining_room"],
            CONF_DURATION: "4:00",
        },
        "priority_inputs": [],
        "expected_result": {
            "errors": {CONF_COLOR: "invalid_color_out_of_range"},
            "description_placeholders": {
                "config_title": "Medicine",
                "color": "521",
            },
        },
    },
    "invalid_duration_string": {
        "user_input": {
            CONF_NAME: "Medicine",
            CONF_COLOR: "cyan",
            CONF_EFFECT: "slow_blink",
            CONF_SWITCH_ENTITIES: ["light.dining_room"],
            CONF_DURATION: "4h",
        },
        "priority_inputs": [],
        "expected_result": {
            "errors": {CONF_DURATION: "invalid_duration"},
            "description_placeholders": {
                "config_title": "Medicine",
                "duration": "4h",
            },
        },
    },
    "invalid_advanced_combo": {
        "user_input": {
            CONF_NAME: "Medicine",
            CONF_COLOR: "cyan",
            CONF_EFFECT: "slow_blink",
            CONF_SWITCH_ENTITIES: ["light.dining_room"],
            CONF_DURATION: "4:00",
            SECTION_ADVANCED_OPTIONS: {
                CONF_LED_CONFIG: [
                    {
                        CONF_COLOR: "red",
                        CONF_EFFECT: "slow_blink",
                        CONF_DURATION: "4:00",
                    },
                    {
                        CONF_COLOR: "orange",
                        CONF_EFFECT: "slow_blink",
                        CONF_DURATION: "4:00",
                    },
                    {
                        CONF_COLOR: "white",
                        CONF_EFFECT: "slow_blink",
                        CONF_DURATION: "4:00",
                    },
                    {
                        CONF_COLOR: "red",
                        CONF_EFFECT: "slow_blink",
                        CONF_DURATION: "4:00",
                    },
                    {
                        CONF_COLOR: "orange",
                        CONF_EFFECT: "slow_blink",
                        CONF_DURATION: "4:00",
                    },
                    {
                        CONF_COLOR: "white",
                        CONF_EFFECT: "slow_blink",
                        CONF_DURATION: "4:00",
                    },
                    {CONF_COLOR: 0, CONF_EFFECT: "slow_blink", CONF_DURATION: "4:00"},
                ]
            },
        },
        "priority_inputs": [],
        "expected_result": {
            "errors": {SECTION_ADVANCED_OPTIONS: "invalid_led_config_override"},
            "description_placeholders": {
                "config_title": "Medicine",
                "key": CONF_COLOR,
            },
        },
    },
    "invalid_led_config_length": {
        "user_input": {
            CONF_NAME: "Medicine",
            CONF_COLOR: "",
            CONF_SWITCH_ENTITIES: ["light.dining_room"],
            SECTION_ADVANCED_OPTIONS: {
                CONF_LED_CONFIG: ["red", "orange", "white", "red", "orange", "white"]
            },
        },
        "priority_inputs": [],
        "expected_result": {
            "errors": {SECTION_ADVANCED_OPTIONS: "invalid_led_config_length"},
            "description_placeholders": {
                "config_title": "Medicine",
            },
        },
    },
    "invalid_led_config_member": {
        "user_input": {
            CONF_NAME: "Medicine",
            CONF_COLOR: "",
            CONF_SWITCH_ENTITIES: ["light.dining_room"],
            SECTION_ADVANCED_OPTIONS: {
                CONF_LED_CONFIG: [
                    "red",
                    "orange",
                    "white",
                    "red",
                    "orange",
                    "white",
                    "pinkish",
                ]
            },
        },
        "priority_inputs": [],
        "expected_result": {
            "errors": {SECTION_ADVANCED_OPTIONS: "invalid_led_config_member"},
            "description_placeholders": {
                "config_title": "Medicine",
                "color": "pinkish",
                "index": "6",
            },
        },
    },
    "invalid_led_config_member_type": {
        "user_input": {
            CONF_NAME: "Medicine",
            CONF_COLOR: "",
            CONF_SWITCH_ENTITIES: ["light.dining_room"],
            SECTION_ADVANCED_OPTIONS: {
                CONF_LED_CONFIG: [
                    "red",
                    "orange",
                    "white",
                    "red",
                    "orange",
                    "white",
                    {CONF_COLOR: {"red": 0}},
                ]
            },
        },
        "priority_inputs": [],
        "expected_result": {
            "errors": {SECTION_ADVANCED_OPTIONS: "invalid_color_type"},
            "description_placeholders": {
                "config_title": "Medicine",
                "type": "dict",
                "index": "6",
            },
        },
    },
    "invalid_led_config_type": {
        "user_input": {
            CONF_NAME: "Medicine",
            CONF_COLOR: "",
            CONF_SWITCH_ENTITIES: ["light.dining_room"],
            SECTION_ADVANCED_OPTIONS: {CONF_LED_CONFIG: {"red": 0}},
        },
        "priority_inputs": [],
        "expected_result": {
            "errors": {SECTION_ADVANCED_OPTIONS: "invalid_led_config_type"},
            "description_placeholders": {
                "config_title": "Medicine",
                "type": "dict",
            },
        },
    },
    "missing_priorities": {
        "user_input": {
            CONF_NAME: "Medicine",
            CONF_COLOR: "cyan",
            CONF_EFFECT: "slow_blink",
            CONF_SWITCH_ENTITIES: ["light.kitchen"],
        },
        "priority_inputs": [
            [
                "doors_open",
                "windows_open",
                "medicine",
            ]
        ],
        "expected_result": {
            "errors": {"base": "extra_priorities"},
            "description_placeholders": {
                "config_title": "Medicine",
                "switch_id": "light.kitchen",
                "switch_name": "Kitchen Light",
                "overlap": "medicine, doors_open",
                "extra_slugs": "windows_open",
            },
        },
    },
    "extra_priorities": {
        "user_input": {
            CONF_NAME: "Medicine",
            CONF_COLOR: "cyan",
            CONF_EFFECT: "slow_blink",
            CONF_SWITCH_ENTITIES: ["light.kitchen"],
        },
        "priority_inputs": [
            [
                "doors_open",
            ]
        ],
        "expected_result": {
            "errors": {"base": "missing_priorities"},
            "description_placeholders": {
                "config_title": "Medicine",
                "switch_id": "light.kitchen",
                "switch_name": "Kitchen Light",
                "overlap": "medicine, doors_open",
                "missing_slugs": "medicine",
            },
        },
    },
}


@Scenario.parametrize(**FLOW_SCENARIOS)
async def test_user_flow(
    hass: HomeAssistant,
    user_input: dict,
    config_entry: MockConfigEntry,
    priority_inputs: list[list[str]],
    expected_result: dict,
) -> None:
    """Test starting a flow by user."""
    user_input = {SECTION_ADVANCED_OPTIONS: {}, **user_input}

    _add_entry_to_hass(hass, config_entry)

    hass.states.async_set(
        "light.kitchen",
        "off",
        {"friendly_name": "Kitchen Light"},
    )

    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    with patch(
        "custom_components.lampie.async_setup_entry",
        return_value=True,
    ) as mock_setup_entry:
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input=user_input,
        )

        for priority_input in priority_inputs:
            result = await hass.config_entries.flow.async_configure(
                result["flow_id"],
                user_input={CONF_PRIORITY: priority_input},
            )

        expected_result = dict(expected_result)
        expected_errors = expected_result.pop("errors", None)
        expected_placeholders = {**expected_result.pop("description_placeholders", {})}
        expected_data = dict(expected_result.pop("data", {}))
        expected_title = expected_data.pop(CONF_NAME, None)

        # title is not set for initial setup (user) flow since the config entry
        # wasn't yet given a name.
        if expected_placeholders and not priority_inputs:
            expected_placeholders.pop("config_title", None)

        if expected_errors:
            assert result["type"] is FlowResultType.FORM
            assert result["errors"] == expected_errors
            assert result["description_placeholders"] == expected_placeholders
        else:
            assert result["type"] is FlowResultType.CREATE_ENTRY
            assert result["data"] == expected_data
            assert result["title"] == expected_title

        await hass.async_block_till_done()

    assert mock_setup_entry.call_count == 0 if expected_errors else 1


@Scenario.parametrize(**FLOW_SCENARIOS)
async def test_options_flow(
    hass: HomeAssistant,
    user_input: dict,
    config_entry: MockConfigEntry,
    priority_inputs: list[list[str]],
    expected_result: dict,
) -> None:
    """Test options flow."""
    user_input = {SECTION_ADVANCED_OPTIONS: {}, **user_input}

    _add_entry_to_hass(hass, config_entry)

    # note: this wouldn't actually be allowed to be in this state. it would have
    # priorities set to work with the `config_entry`.
    entry_data = {CONF_SWITCH_ENTITIES: user_input[CONF_SWITCH_ENTITIES]}
    advanced_options = user_input.get(SECTION_ADVANCED_OPTIONS, {})
    led_config = advanced_options.get(CONF_LED_CONFIG)

    if led_config:
        entry_data[CONF_LED_CONFIG] = led_config

    target_config_entry = _add_entry_to_hass(
        hass,
        MockConfigEntry(
            domain=DOMAIN,
            title=user_input.pop(CONF_NAME),
            data=entry_data,
            unique_id=f"{DOMAIN}_arbitrary_id",
        ),
    )

    hass.states.async_set(
        "light.kitchen",
        "off",
        {"friendly_name": "Kitchen Light"},
    )

    with patch(
        "custom_components.lampie.async_setup_entry",
        return_value=True,
    ) as mock_setup_entry:
        await hass.config_entries.async_setup(target_config_entry.entry_id)
        await hass.async_block_till_done()
        assert mock_setup_entry.called

        result = await hass.config_entries.options.async_init(
            target_config_entry.entry_id
        )

        assert result["type"] is FlowResultType.FORM
        assert result["step_id"] == "init"

        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input=user_input,
        )

        for priority_input in priority_inputs:
            result = await hass.config_entries.options.async_configure(
                result["flow_id"],
                user_input={CONF_PRIORITY: priority_input},
            )

        expected_errors = expected_result.pop("errors", None)
        expected_placeholders = expected_result.pop("description_placeholders", None)
        expected_data = expected_result.pop("data", {})
        expected_title = expected_data.pop(CONF_NAME, None)

        if expected_errors:
            assert result["type"] is FlowResultType.FORM
            assert result["errors"] == expected_errors
            assert result["description_placeholders"] == expected_placeholders
        else:
            assert result["type"] is FlowResultType.CREATE_ENTRY
            assert result["title"] == expected_title
            assert target_config_entry.data == expected_data

        await hass.async_block_till_done()


async def test_included_switch_entities(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    device_registry: dr.DeviceRegistry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test user flow shows the proper list of entities."""
    _add_entry_to_hass(hass, config_entry)

    add_mock_switch(
        hass, "light.dining_room", {"manufacturer": "Inovelli", "model": "VZM31-SN"}
    )
    add_mock_switch(
        hass, "light.entryway", {"manufacturer": "Aqara", "model": "WS-USC03"}
    )
    add_mock_switch(
        hass, "fan.bedroom", {"manufacturer": "Inovelli", "model": "VZM35-SN"}
    )

    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    schema = result["data_schema"].schema
    switch_entities = schema[CONF_SWITCH_ENTITIES]

    assert isinstance(switch_entities, selector.EntitySelector)
    assert switch_entities.config["domain"] == ["light", "fan"]
    assert (
        switch_entities.config["include_entities"]
        == [
            "light.dining_room",
            "switch.dining_room_local_protection",  # filtered by config.domain
            "switch.dining_room_disable_config_2x_tap_to_clear_notifications",  # filtered by config.domain
            "fan.bedroom",
            "switch.bedroom_local_protection",  # filtered by config.domain
            "switch.bedroom_disable_config_2x_tap_to_clear_notifications",  # filtered by config.domain
        ]
    )


async def test_abort_duplicated_entry(
    hass: HomeAssistant,
) -> None:
    """Test if we abort on duplicate user input data."""
    data = {
        CONF_SWITCH_ENTITIES: ["light.kitchen"],
        CONF_COLOR: "red",
        CONF_EFFECT: "open_close",
    }
    _add_entry_to_hass(
        hass,
        MockConfigEntry(
            domain=DOMAIN,
            title="Doors Open",
            data=data,
        ),
    )

    hass.states.async_set(
        "light.kitchen",
        "off",
        {"friendly_name": "Kitchen Light"},
    )

    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": SOURCE_USER},
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        user_input={CONF_NAME: "Doors Open", **data, SECTION_ADVANCED_OPTIONS: {}},
    )

    with patch("custom_components.lampie.async_setup_entry", return_value=True):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input={
                CONF_PRIORITY: ["doors_open", "doors_open_2"],
            },
        )
        assert result["type"] is FlowResultType.ABORT
        assert result["reason"] == "already_configured"

        await hass.async_block_till_done()


async def test_avoid_duplicated_title(hass: HomeAssistant) -> None:
    """Test if we avoid duplicate titles."""
    _add_entry_to_hass(
        hass,
        MockConfigEntry(
            domain=DOMAIN,
            title="Doors Open",
            data={
                CONF_SWITCH_ENTITIES: ["light.kitchen1"],
                CONF_COLOR: "red",
                CONF_EFFECT: "open_close",
            },
        ),
    )

    _add_entry_to_hass(
        hass,
        MockConfigEntry(
            domain=DOMAIN,
            title="Doors Open 3",
            data={
                CONF_SWITCH_ENTITIES: ["light.kitchen3"],
                CONF_COLOR: "red",
                CONF_EFFECT: "open_close",
            },
        ),
    )

    for i in range(4):
        hass.states.async_set(
            f"light.kitchen{i + 1}",
            "off",
            {"friendly_name": "Kitchen Light"},
        )

    with patch("custom_components.lampie.async_setup_entry", return_value=True):
        result = await hass.config_entries.flow.async_init(
            DOMAIN,
            context={"source": SOURCE_USER},
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input={
                CONF_NAME: "Doors Open",
                CONF_SWITCH_ENTITIES: ["light.kitchen2"],
                CONF_COLOR: "red",
                CONF_EFFECT: "open_close",
                SECTION_ADVANCED_OPTIONS: {},
            },
        )
        assert result["type"] is FlowResultType.CREATE_ENTRY
        assert result["title"] == "Doors Open 2"

        await hass.async_block_till_done()

        result = await hass.config_entries.flow.async_init(
            DOMAIN,
            context={"source": SOURCE_USER},
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input={
                CONF_NAME: "Doors Open",
                CONF_SWITCH_ENTITIES: ["light.kitchen4"],
                CONF_COLOR: "red",
                CONF_EFFECT: "open_close",
                SECTION_ADVANCED_OPTIONS: {},
            },
        )
        assert result["type"] is FlowResultType.CREATE_ENTRY
        assert result["title"] == "Doors Open 4"

        await hass.async_block_till_done()


async def test_update_priorities_across_entries(hass: HomeAssistant) -> None:
    """Test if we update priorities across all entries."""
    doors_open_entry = _add_entry_to_hass(
        hass,
        MockConfigEntry(
            domain=DOMAIN,
            title="Doors Open",
            data={
                CONF_SWITCH_ENTITIES: ["light.kitchen", "light.entryway"],
                CONF_COLOR: "red",
                CONF_EFFECT: "open_close",
                CONF_PRIORITY: {
                    "light.kitchen": [
                        "doors_open",
                        "medicine",
                    ],
                },
            },
        ),
    )
    await setup_added_integration(hass, doors_open_entry)

    windows_open_entry = _add_entry_to_hass(
        hass,
        MockConfigEntry(
            domain=DOMAIN,
            title="Windows Open",
            data={
                CONF_SWITCH_ENTITIES: ["light.dining_room", "light.bedroom"],
                CONF_COLOR: "orange",
                CONF_EFFECT: "open_close",
                CONF_PRIORITY: {
                    "light.bedroom": [
                        "windows_open",
                        "medicine",
                    ],
                    "light.dining_room": [
                        "medicine",
                        "windows_open",
                    ],
                },
            },
        ),
    )
    await setup_added_integration(hass, windows_open_entry)

    medicine_entry = _add_entry_to_hass(
        hass,
        MockConfigEntry(
            domain=DOMAIN,
            title="Medicine",
            disabled_by=ConfigEntryDisabler.USER,
            data={
                CONF_SWITCH_ENTITIES: [
                    "light.kitchen",
                    "light.dining_room",
                    "light.bedroom",
                ],
                CONF_COLOR: "cyan",
                CONF_EFFECT: "slow_blink",
                CONF_PRIORITY: {
                    "light.bedroom": [
                        "windows_open",
                        "medicine",
                    ],
                    "light.dining_room": [
                        "medicine",
                        "windows_open",
                    ],
                    "light.kitchen": [
                        "doors_open",
                        "medicine",
                    ],
                },
            },
        ),
    )
    await setup_added_integration(hass, medicine_entry)

    unrelated_entry = _add_entry_to_hass(
        hass,
        MockConfigEntry(
            domain=DOMAIN,
            title="Unrelated",
            data={
                CONF_SWITCH_ENTITIES: [
                    "light.bathroom",
                ],
                CONF_COLOR: "purple",
                CONF_EFFECT: "aurora",
            },
        ),
    )
    await setup_added_integration(hass, unrelated_entry)

    hass.states.async_set(
        "light.kitchen",
        "off",
        {"friendly_name": "Kitchen Light"},
    )
    hass.states.async_set(
        "light.dining_room",
        "off",
        {"friendly_name": "Dining Room Light"},
    )
    hass.states.async_set(
        "light.entryway",
        "off",
        {"friendly_name": "Entryway Light"},
    )
    hass.states.async_set(
        "light.bedroom",
        "off",
        {"friendly_name": "Bedroom Light"},
    )

    assert doors_open_entry.state is ConfigEntryState.LOADED
    assert windows_open_entry.state is ConfigEntryState.LOADED
    assert medicine_entry.state is ConfigEntryState.NOT_LOADED
    assert unrelated_entry.state is ConfigEntryState.LOADED

    async def reconfigure_windows_open(color: str):
        # start the options flow for the windows open entry
        result = await hass.config_entries.options.async_init(
            windows_open_entry.entry_id
        )

        assert result["type"] is FlowResultType.FORM
        assert result["step_id"] == "init"

        # changes to the the windows open entry:
        #   - add light.entryway switch
        #   - remove light.kitchen switch
        #   - remove light.bedroom switch
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={
                CONF_SWITCH_ENTITIES: ["light.dining_room", "light.entryway"],
                CONF_COLOR: color,
                CONF_EFFECT: "open_close",
                SECTION_ADVANCED_OPTIONS: {},
            },
        )

        # setup for light.dining_room priority in the the windows open entry;
        # this just stays the same, but is still part of the flow
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={
                CONF_PRIORITY: [
                    "medicine",
                    "windows_open",
                ]
            },
        )

        # setup for light.entryway priority in the the windows open entry;
        # this is new because light.entryway was added
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={
                CONF_PRIORITY: [
                    "doors_open",
                    "windows_open",
                ]
            },
        )

        await hass.async_block_till_done()
        assert result["type"] is FlowResultType.CREATE_ENTRY

        assert doors_open_entry.state is ConfigEntryState.LOADED
        assert doors_open_entry.data[CONF_PRIORITY] == {
            "light.kitchen": [
                "doors_open",
                "medicine",
            ],
            "light.entryway": [
                "doors_open",
                "windows_open",
            ],
        }
        assert windows_open_entry.state is ConfigEntryState.LOADED
        assert windows_open_entry.data[CONF_PRIORITY] == {
            "light.dining_room": [
                "medicine",
                "windows_open",
            ],
            "light.entryway": [
                "doors_open",
                "windows_open",
            ],
        }
        assert medicine_entry.state is ConfigEntryState.NOT_LOADED
        assert medicine_entry.data[CONF_PRIORITY] == {
            "light.dining_room": [
                "medicine",
                "windows_open",
            ],
            "light.kitchen": [
                "doors_open",
                "medicine",
            ],
        }
        assert unrelated_entry.state is ConfigEntryState.LOADED

    def call_slugs(mock_calls) -> set[str]:
        entry_slugs = {
            doors_open_entry.entry_id: "doors_open",
            windows_open_entry.entry_id: "windows_open",
            medicine_entry.entry_id: "medicine",
            unrelated_entry.entry_id: "unrelated",
        }
        return {
            (
                (entry_id := arg.entry_id if isinstance(arg, ConfigEntry) else arg)
                and entry_slugs.get(entry_id)
            )
            or arg
            for call in mock_calls
            for arg in call.args
        }

    @contextmanager
    def patches():
        with (
            patch("custom_components.lampie.async_setup_entry", return_value=True),
            patch.object(
                hass.config_entries,
                "async_update_entry",
                side_effect=hass.config_entries.async_update_entry,
            ) as mock_update_entry,
            patch.object(
                hass.config_entries,
                "async_reload",
                return_value=True,
            ) as mock_reload,
            patch.object(
                hass.config_entries,
                "async_unload",
                side_effect=hass.config_entries.async_unload,
            ) as mock_unload,
            patch.object(
                hass.config_entries,
                "async_setup",
                side_effect=hass.config_entries.async_setup,
            ) as mock_setup,
        ):
            yield {
                "update_entry": mock_update_entry,
                "reload": mock_reload,
                "unload": mock_unload,
                "setup": mock_setup,
            }

    with patches() as mocks:
        await reconfigure_windows_open("yellow")

        assert call_slugs(mocks["update_entry"].mock_calls) == {
            "doors_open",
            "medicine",
            "windows_open",
        }, "update_entry calls do not match expected"

        assert call_slugs(mocks["unload"].mock_calls) == {
            "doors_open",
        }, "unload calls do not match expected"

        assert call_slugs(mocks["setup"].mock_calls) == {
            "doors_open",
        }, "setup calls do not match expected"

    # another run through should not result in unload or setup at all because
    # there are no changes to the priorities.
    with patches() as mocks:
        await reconfigure_windows_open("green")

        assert call_slugs(mocks["update_entry"].mock_calls) == {
            "windows_open",
        }, "update_entry calls do not match expected (no priority change)"

        assert call_slugs(mocks["unload"].mock_calls) == set(), (
            "mock_unload calls do not match expected (no priority change)"
        )
        assert call_slugs(mocks["setup"].mock_calls) == set(), (
            "mock_setup calls do not match expected (no priority change)"
        )


def _add_entry_to_hass(hass: HomeAssistant, entry: MockConfigEntry):
    entry.runtime_data = LampieConfigEntryRuntimeData(
        orchestrator=None,  # type: ignore[arg-type]
        coordinator=None,  # type: ignore[arg-type]
    )
    entry.add_to_hass(hass)
    return entry
