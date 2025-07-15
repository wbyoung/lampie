"""Test component setup."""

import datetime as dt
import logging
from typing import Any
from unittest.mock import patch

from homeassistant.components.switch import DOMAIN as SWITCH_DOMAIN
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import ATTR_ENTITY_ID, SERVICE_TURN_OFF, SERVICE_TURN_ON
from homeassistant.core import Event, HomeAssistant
from homeassistant.helpers import (
    device_registry as dr,
    entity_registry as er,
    issue_registry as ir,
)
from homeassistant.setup import async_setup_component
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_mock_service,
)
from syrupy.assertion import SnapshotAssertion

from custom_components.lampie.const import (
    ATTR_COLOR,
    ATTR_DURATION,
    ATTR_LED_CONFIG,
    ATTR_NAME,
    CONF_COLOR,
    CONF_DURATION,
    CONF_EFFECT,
    CONF_END_ACTION,
    CONF_LED_CONFIG,
    CONF_PRIORITY,
    CONF_START_ACTION,
    CONF_SWITCH_ENTITIES,
    DOMAIN,
    TRACE,
)
from custom_components.lampie.services import (
    SERVICE_NAME_ACTIVATE,
    SERVICE_NAME_OVERRIDE,
)
from custom_components.lampie.types import (
    Color,
    Effect,
    LEDConfig,
    LEDConfigSource,
    LEDConfigSourceType,
)

from . import MockNow, Scenario, add_mock_switch, setup_integration
from .syrupy import any_device_id_matcher

_LOGGER = logging.getLogger(__name__)


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
    assert device == snapshot(name="device")


def _response_script(name: str, response: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "sequence": [
                {
                    "variables": {"lampie_response": response},
                },
                {
                    "stop": "done",
                    "response_variable": "lampie_response",
                },
            ]
        },
    }


@Scenario.parametrize(
    Scenario(
        "doors_open_10s_duration_expired",
        {
            "configs": {
                "doors_open": {CONF_DURATION: dt.timedelta(seconds=10).total_seconds()}
            },
            "steps": [
                {
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                    "target": "switch.doors_open_notification",
                },
                {"event": {"command": "led_effect_complete_ALL_LEDS"}},
            ],
            "expected_notification_state": "off",
            "expected_notifiation_timer": False,
            "expected_switch_timer": False,
            "expected_events": 1,
            "expected_zha_calls": 1,
        },
    ),
    Scenario(
        "doors_open_5m_duration_unexpired",
        {
            "configs": {
                "doors_open": {CONF_DURATION: dt.timedelta(minutes=5).total_seconds()}
            },
            "steps": [
                {
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                    "target": "switch.doors_open_notification",
                },
            ],
            "expected_notification_state": "on",
            "expected_notifiation_timer": True,
            "expected_switch_timer": False,
            "expected_events": 0,
            "expected_zha_calls": 1,
        },
    ),
    Scenario(
        "doors_open_5m_duration_expired",
        {
            "configs": {
                "doors_open": {CONF_DURATION: dt.timedelta(minutes=5).total_seconds()}
            },
            "steps": [
                {
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                    "target": "switch.doors_open_notification",
                },
                {"delay": dt.timedelta(minutes=5, seconds=1)},
            ],
            "expected_notification_state": "off",
            "expected_notifiation_timer": False,
            "expected_switch_timer": False,
            "expected_events": 1,
            "expected_zha_calls": 2,
        },
    ),
    Scenario(
        "doors_open_mixed_long_durations_partially_expired",
        {
            "configs": {
                "doors_open": {
                    CONF_LED_CONFIG: [
                        {CONF_COLOR: "red"},
                        {CONF_COLOR: "orange", CONF_DURATION: "0:05"},
                        {CONF_COLOR: "white", CONF_DURATION: "0:10"},
                        {CONF_COLOR: "red"},
                        {CONF_COLOR: "orange", CONF_DURATION: "0:05"},
                        {CONF_COLOR: "white", CONF_DURATION: "0:10"},
                        {CONF_COLOR: 0},
                    ],
                }
            },
            "steps": [
                {
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                    "target": "switch.doors_open_notification",
                },
                {"delay": dt.timedelta(minutes=5, seconds=1)},
            ],
            "expected_notification_state": "on",
            "expected_notifiation_timer": True,
            "expected_switch_timer": False,
            "expected_events": 0,
            "expected_zha_calls": 9,
        },
    ),
    Scenario(
        "doors_open_mixed_long_durations_expired",
        {
            "configs": {
                "doors_open": {
                    CONF_LED_CONFIG: [
                        {CONF_COLOR: "red", CONF_DURATION: "0:05"},
                        {CONF_COLOR: "orange", CONF_DURATION: "0:05"},
                        {CONF_COLOR: "white", CONF_DURATION: "0:10"},
                        {CONF_COLOR: "red", CONF_DURATION: "0:05"},
                        {CONF_COLOR: "orange", CONF_DURATION: "0:05"},
                        {CONF_COLOR: "white", CONF_DURATION: "0:10"},
                        {CONF_COLOR: 0, CONF_DURATION: "0:05"},
                    ],
                }
            },
            "steps": [
                {
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                    "target": "switch.doors_open_notification",
                },
                {"delay": dt.timedelta(minutes=5, seconds=1)},
                {"delay": dt.timedelta(minutes=5, seconds=1)},
            ],
            "expected_notification_state": "off",
            "expected_notifiation_timer": False,
            "expected_switch_timer": False,
            "expected_events": 1,
            "expected_zha_calls": 14,
        },
    ),
    Scenario(
        "doors_open_5m_dismissed",
        {
            "configs": {
                "doors_open": {CONF_DURATION: dt.timedelta(minutes=5).total_seconds()}
            },
            "steps": [
                {
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                    "target": "switch.doors_open_notification",
                },
                {"event": {"command": "button_3_double"}},
            ],
            "expected_notification_state": "off",
            "expected_notifiation_timer": False,
            "expected_switch_timer": False,
            "expected_events": 1,
            "expected_zha_calls": 1,
        },
    ),
    Scenario(
        "doors_open_5m_turned_off",
        {
            "configs": {
                "doors_open": {CONF_DURATION: dt.timedelta(minutes=5).total_seconds()}
            },
            "steps": [
                {
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                    "target": "switch.doors_open_notification",
                },
                {
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_OFF}",
                    "target": "switch.doors_open_notification",
                },
            ],
            "expected_notification_state": "off",
            "expected_notifiation_timer": False,
            "expected_switch_timer": False,
            "expected_events": 0,
            "expected_zha_calls": 2,
        },
    ),
    Scenario(
        "doors_open_5m_reactivated_and_unexpired",
        {
            "configs": {
                "doors_open": {CONF_DURATION: dt.timedelta(minutes=5).total_seconds()}
            },
            "steps": [
                {
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                    "target": "switch.doors_open_notification",
                },
                {
                    "action": f"{DOMAIN}.{SERVICE_NAME_ACTIVATE}",
                    "data": {
                        "notification": "doors_open",
                    },
                },
            ],
            "expected_notification_state": "on",
            "expected_notifiation_timer": True,
            "expected_switch_timer": False,
            "expected_events": 0,
            "expected_zha_calls": 1,
        },
    ),
    Scenario(
        "doors_open_5m_duration_expired_not_blocked_via_end_action",
        {
            "configs": {
                "doors_open": {
                    CONF_DURATION: dt.timedelta(minutes=5).total_seconds(),
                    CONF_END_ACTION: "script.block_dismissal",
                }
            },
            "scripts": _response_script("block_dismissal", {"block_dismissal": True}),
            "steps": [
                {
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                    "target": "switch.doors_open_notification",
                },
                {"delay": dt.timedelta(minutes=5, seconds=1)},
            ],
            "expected_notification_state": "off",
            "expected_notifiation_timer": False,
            "expected_switch_timer": False,
            "expected_events": 1,
            "expected_zha_calls": 2,
        },
    ),
    Scenario(
        "doors_open_multiple_switches_dismissed",
        {
            "configs": {
                "doors_open": {
                    CONF_SWITCH_ENTITIES: ["light.kitchen", "light.entryway"],
                }
            },
            "initial_states": {
                "light.entryway": "on",
            },
            "steps": [
                {
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                    "target": "switch.doors_open_notification",
                },
                {"event": {"command": "button_3_double"}},
            ],
            "expected_notification_state": "off",
            "expected_notifiation_timer": False,
            "expected_switch_timer": False,
            "expected_events": 1,
            "expected_zha_calls": 3,
        },
    ),
    Scenario(
        "kitchen_override_leds_named",
        {
            "configs": {},
            "steps": [
                {
                    "action": f"{DOMAIN}.{SERVICE_NAME_OVERRIDE}",
                    "target": "light.kitchen",
                    "data": {
                        ATTR_NAME: "customized_name",
                        ATTR_LED_CONFIG: [
                            {ATTR_COLOR: "green"},
                        ],
                    },
                },
            ],
            "expected_notification_state": "off",
            "expected_notifiation_timer": False,
            "expected_switch_timer": False,
            "expected_events": 0,
            "expected_zha_calls": 1,
        },
    ),
    Scenario(
        "kitchen_override_leds_clear",
        {
            "configs": {},
            "steps": [
                {
                    "action": f"{DOMAIN}.{SERVICE_NAME_OVERRIDE}",
                    "target": "light.kitchen",
                    "data": {
                        ATTR_LED_CONFIG: [
                            {ATTR_COLOR: "green"},
                        ],
                    },
                },
                {
                    "action": f"{DOMAIN}.{SERVICE_NAME_OVERRIDE}",
                    "target": "light.kitchen",
                    "data": {
                        ATTR_LED_CONFIG: [],
                    },
                },
            ],
            "expected_notification_state": "off",
            "expected_notifiation_timer": False,
            "expected_switch_timer": False,
            "expected_events": 0,
            "expected_zha_calls": 2,
        },
    ),
    Scenario(
        "kitchen_override_leds_reset",
        {
            "configs": {},
            "steps": [
                {
                    "action": f"{DOMAIN}.{SERVICE_NAME_OVERRIDE}",
                    "target": "light.kitchen",
                    "data": {
                        ATTR_LED_CONFIG: [
                            {ATTR_COLOR: "green"},
                        ],
                    },
                },
                {
                    "action": f"{DOMAIN}.{SERVICE_NAME_OVERRIDE}",
                    "target": "light.kitchen",
                    "data": {
                        ATTR_LED_CONFIG: None,
                    },
                },
            ],
            "expected_notification_state": "off",
            "expected_notifiation_timer": False,
            "expected_switch_timer": False,
            "expected_events": 0,
            "expected_zha_calls": 2,
        },
    ),
    Scenario(
        "kitchen_override_leds_dismissed",
        {
            "configs": {},
            "steps": [
                {
                    "action": f"{DOMAIN}.{SERVICE_NAME_OVERRIDE}",
                    "target": "light.kitchen",
                    "data": {
                        ATTR_LED_CONFIG: [
                            {ATTR_COLOR: "green"},
                        ],
                    },
                },
                {"event": {"command": "button_3_double"}},
            ],
            "expected_notification_state": "off",
            "expected_notifiation_timer": False,
            "expected_switch_timer": False,
            "expected_events": 1,
            "expected_zha_calls": 1,
        },
    ),
    Scenario(
        "kitchen_override_leds_5m_duration_unexpired",
        {
            "configs": {},
            "steps": [
                {
                    "action": f"{DOMAIN}.{SERVICE_NAME_OVERRIDE}",
                    "target": "light.kitchen",
                    "data": {
                        ATTR_LED_CONFIG: [
                            {ATTR_COLOR: "green", ATTR_DURATION: "0:05"},
                        ],
                    },
                },
            ],
            "expected_notification_state": "off",
            "expected_notifiation_timer": False,
            "expected_switch_timer": True,
            "expected_events": 0,
            "expected_zha_calls": 1,
        },
    ),
    Scenario(
        "kitchen_override,doors_open_on",
        {
            "configs": {},
            "steps": [
                {
                    "action": f"{DOMAIN}.{SERVICE_NAME_OVERRIDE}",
                    "target": "light.kitchen",
                    "data": {
                        ATTR_LED_CONFIG: [
                            {ATTR_COLOR: "green"},
                        ],
                    },
                },
                {
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                    "target": "switch.doors_open_notification",
                },
            ],
            "expected_notification_state": "on",
            "expected_notifiation_timer": False,
            "expected_switch_timer": False,
            "expected_events": 0,
            "expected_zha_calls": 1,
        },
    ),
    Scenario(
        "doors_open_on,kitchen_override_leds_dismissed",
        {
            "configs": {},
            "steps": [
                {
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                    "target": "switch.doors_open_notification",
                },
                {
                    "action": f"{DOMAIN}.{SERVICE_NAME_OVERRIDE}",
                    "target": "light.kitchen",
                    "data": {
                        ATTR_LED_CONFIG: [
                            {ATTR_COLOR: "green"},
                        ],
                    },
                },
                {"event": {"command": "button_3_double"}},
            ],
            "expected_notification_state": "on",
            "expected_notifiation_timer": False,
            "expected_switch_timer": False,
            "expected_events": 1,
            "expected_zha_calls": 2,
        },
    ),
    Scenario(
        "doors_open_on,kitchen_override_leds_5m_duration_expired",
        {
            "configs": {},
            "steps": [
                {
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                    "target": "switch.doors_open_notification",
                },
                {
                    "action": f"{DOMAIN}.{SERVICE_NAME_OVERRIDE}",
                    "target": "light.kitchen",
                    "data": {
                        ATTR_LED_CONFIG: [
                            {ATTR_COLOR: "green", ATTR_DURATION: "0:05"},
                        ],
                    },
                },
                {"delay": dt.timedelta(minutes=5, seconds=1)},
            ],
            "expected_notification_state": "on",
            "expected_notifiation_timer": False,
            "expected_switch_timer": False,
            "expected_events": 1,
            "expected_zha_calls": 3,
        },
    ),
    Scenario(
        "kitchen_override_leds_5m_duration_expired",
        {
            "configs": {},
            "steps": [
                {
                    "action": f"{DOMAIN}.{SERVICE_NAME_OVERRIDE}",
                    "target": "light.kitchen",
                    "data": {
                        ATTR_LED_CONFIG: [
                            {ATTR_COLOR: "green", ATTR_DURATION: "0:05"},
                        ],
                    },
                },
                {"delay": dt.timedelta(minutes=5, seconds=1)},
            ],
            "expected_notification_state": "off",
            "expected_notifiation_timer": False,
            "expected_switch_timer": False,
            "expected_events": 1,
            "expected_zha_calls": 2,
        },
    ),
    Scenario(
        "kitchen_override_leds_mixed_long_durations_partially_expired",
        {
            "configs": {},
            "steps": [
                {
                    "action": f"{DOMAIN}.{SERVICE_NAME_OVERRIDE}",
                    "target": "light.kitchen",
                    "data": {
                        ATTR_LED_CONFIG: [
                            {CONF_COLOR: "red", CONF_DURATION: "0:10"},
                            {CONF_COLOR: "orange", CONF_DURATION: "0:05"},
                            {CONF_COLOR: "white", CONF_DURATION: "0:10"},
                            {CONF_COLOR: "red", CONF_DURATION: "0:10"},
                            {CONF_COLOR: "orange", CONF_DURATION: "0:05"},
                            {CONF_COLOR: "white", CONF_DURATION: "0:10"},
                            {CONF_COLOR: 0, CONF_DURATION: "0:10"},
                        ],
                    },
                },
                {"delay": dt.timedelta(minutes=5, seconds=1)},
            ],
            "expected_notification_state": "off",
            "expected_notifiation_timer": False,
            "expected_switch_timer": True,
            "expected_events": 0,
            "expected_zha_calls": 9,
        },
    ),
    Scenario(
        "kitchen_override_leds_mixed_long_durations_expired",
        {
            "configs": {},
            "steps": [
                {
                    "action": f"{DOMAIN}.{SERVICE_NAME_OVERRIDE}",
                    "target": "light.kitchen",
                    "data": {
                        ATTR_LED_CONFIG: [
                            {CONF_COLOR: "red", CONF_DURATION: "0:05"},
                            {CONF_COLOR: "orange", CONF_DURATION: "0:05"},
                            {CONF_COLOR: "white", CONF_DURATION: "0:10"},
                            {CONF_COLOR: "red", CONF_DURATION: "0:05"},
                            {CONF_COLOR: "orange", CONF_DURATION: "0:05"},
                            {CONF_COLOR: "white", CONF_DURATION: "0:10"},
                            {CONF_COLOR: 0, CONF_DURATION: "0:05"},
                        ],
                    },
                },
                {"delay": dt.timedelta(minutes=5, seconds=1)},
                {"delay": dt.timedelta(minutes=5, seconds=1)},
            ],
            "expected_notification_state": "off",
            "expected_notifiation_timer": False,
            "expected_switch_timer": False,
            "expected_events": 1,
            "expected_zha_calls": 14,
        },
    ),
    Scenario(
        "kitchen_override_leds_5m_duration_dismissed",
        {
            "configs": {},
            "steps": [
                {
                    "action": f"{DOMAIN}.{SERVICE_NAME_OVERRIDE}",
                    "target": "light.kitchen",
                    "data": {
                        ATTR_LED_CONFIG: [
                            {ATTR_COLOR: "green", ATTR_DURATION: "0:05"},
                        ],
                    },
                },
                {"event": {"command": "button_3_double"}},
            ],
            "expected_notification_state": "off",
            "expected_notifiation_timer": False,
            "expected_switch_timer": False,
            "expected_events": 1,
            "expected_zha_calls": 1,
        },
    ),
    Scenario(
        "entryway_override",
        {
            "configs": {},
            "initial_states": {
                "light.entryway": "on",
            },
            "steps": [
                {
                    "action": f"{DOMAIN}.{SERVICE_NAME_OVERRIDE}",
                    "target": "light.entryway",
                    "data": {
                        ATTR_LED_CONFIG: [
                            {ATTR_COLOR: "green"},
                        ],
                    },
                },
            ],
            "expected_notification_state": "off",
            "expected_notifiation_timer": False,
            "expected_switch_timer": False,
            "expected_events": 0,
            "expected_zha_calls": 1,
        },
    ),
)
async def test_toggle_notifications(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    switch: er.RegistryEntry,
    now: MockNow,
    configs: dict[str, dict[str, Any]],
    initial_states: dict[str, str],
    scripts: dict[str, Any],
    steps: list[dict[str, Any]],
    expected_notification_state: str,
    expected_notifiation_timer: bool,
    expected_switch_timer: bool,
    expected_events: int,
    expected_zha_calls: int,
    device_registry: dr.DeviceRegistry,
    entity_registry: er.EntityRegistry,
    snapshot: SnapshotAssertion,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test turning on and off a notification."""
    await setup_integration(hass, config_entry)

    hass.config_entries.async_update_entry(
        config_entry,
        data={
            **config_entry.data,
            **(configs or {}).get("doors_open", {}),
        },
    )

    for entity_id, state in (initial_states or {}).items():
        add_mock_switch(hass, entity_id)
        hass.states.async_set(entity_id, state)

    assert await async_setup_component(hass, "script", {"script": scripts})

    cluster_commands = async_mock_service(hass, "zha", "issue_zigbee_cluster_command")
    async_call = hass.services.async_call
    events: list[Event] = []

    def capture_events(event: Event) -> None:
        events.append(event)

    hass.bus.async_listen(
        "lampie.dismissed",
        capture_events,
    )
    hass.bus.async_listen(
        "lampie.expired",
        capture_events,
    )

    with patch(
        "homeassistant.core.ServiceRegistry.async_call",
        side_effect=async_call,
    ) as mocked_service_call:
        for step in steps:
            _LOGGER.log(TRACE, "step %r", step)

            if "action" in step:
                domain, service_name = step["action"].split(".")
                args = {**step.get("data", {})}
                if "target" in step:
                    args[ATTR_ENTITY_ID] = step["target"]
                await async_call(
                    domain,
                    service_name,
                    args,
                    blocking=True,
                )
            elif "event" in step:
                hass.bus.async_fire(
                    "zha_event",
                    {
                        "device_id": switch.device_id,
                        **step["event"],
                    },
                )
            elif "delay" in step:
                now._tick(step["delay"].total_seconds())
            await hass.async_block_till_done()

        # capture service calls to assert about start/end action invocations
        service_calls = [
            (domain, service, args, *rest)
            for call in mocked_service_call.mock_calls
            for domain, service, args, *rest in [call.args]
            if (domain, service) != ("zha", "issue_zigbee_cluster_command")
        ]

    assert (
        hass.states.get("switch.doors_open_notification").state
        == expected_notification_state
    )

    orchestrator = config_entry.runtime_data.orchestrator
    notification_info = orchestrator.notification_info("doors_open")
    switch_info = orchestrator.switch_info(switch.entity_id)

    assert notification_info == snapshot(name="notification_info")
    assert switch_info == snapshot(name="switch_info")
    assert (
        bool(notification_info.expiration.cancel_listener) == expected_notifiation_timer
    )
    assert bool(switch_info.expiration.cancel_listener) == expected_switch_timer

    if expected_zha_calls:
        assert cluster_commands == snapshot(name="zha_cluster_commands")
    assert len(cluster_commands) == expected_zha_calls

    if expected_events:
        assert events == snapshot(name="events")
    assert len(events) == expected_events

    assert service_calls == snapshot(
        name="service_calls", matcher=any_device_id_matcher
    )

    for entity in entity_registry.entities.get_entries_for_config_entry_id(
        config_entry.entry_id
    ):
        assert hass.states.get(entity.entity_id) == snapshot(name=entity.entity_id)


@Scenario.parametrize(
    Scenario(
        "doors_open_on",
        {
            "configs": {},
            "steps": [
                {
                    "target": "switch.doors_open_notification",
                    "action": SERVICE_TURN_ON,
                }
            ],
        },
    ),
    Scenario(
        "medicine_on",
        {
            "configs": {},
            "steps": [
                {
                    "target": "switch.medicine_notification",
                    "action": SERVICE_TURN_ON,
                }
            ],
        },
    ),
    Scenario(
        "both_on",
        {
            "configs": {},
            "steps": [
                {
                    "target": "switch.doors_open_notification",
                    "action": SERVICE_TURN_ON,
                },
                {
                    "target": "switch.medicine_notification",
                    "action": SERVICE_TURN_ON,
                },
            ],
        },
    ),
    Scenario(
        "doors_open_on_with_third_switch",
        {
            "configs": {
                "doors_open": {
                    CONF_SWITCH_ENTITIES: ["light.entryway", "light.dining_room"],
                    CONF_PRIORITY: {
                        "light.dining_room": ["medicine", "doors_open"],
                    },
                },
                "medicine": {
                    CONF_SWITCH_ENTITIES: ["light.dining_room", "light.kitchen"],
                    CONF_PRIORITY: {
                        "light.dining_room": ["medicine", "doors_open"],
                    },
                },
            },
            "steps": [
                {
                    "target": "switch.doors_open_notification",
                    "action": SERVICE_TURN_ON,
                },
            ],
        },
    ),
    Scenario(
        "both_on_then_medicine_off",
        {
            "configs": {},
            "steps": [
                {
                    "target": "switch.doors_open_notification",
                    "action": SERVICE_TURN_ON,
                },
                {
                    "target": "switch.medicine_notification",
                    "action": SERVICE_TURN_ON,
                },
                {
                    "target": "switch.medicine_notification",
                    "action": SERVICE_TURN_OFF,
                },
            ],
        },
    ),
    Scenario(
        "both_on_then_doors_open_off",
        {
            "configs": {},
            "steps": [
                {
                    "target": "switch.doors_open_notification",
                    "action": SERVICE_TURN_ON,
                },
                {
                    "target": "switch.medicine_notification",
                    "action": SERVICE_TURN_ON,
                },
                {
                    "target": "switch.doors_open_notification",
                    "action": SERVICE_TURN_OFF,
                },
            ],
        },
    ),
    Scenario(
        "both_on_then_doors_open_off_custom_medicine_leds",
        {
            "configs": {
                "medicine": {
                    CONF_LED_CONFIG: [
                        {CONF_COLOR: "red"},
                        {CONF_COLOR: "orange"},
                        {CONF_COLOR: "white"},
                        {CONF_COLOR: "red"},
                        {CONF_COLOR: "orange"},
                        {CONF_COLOR: "white"},
                        {CONF_COLOR: 0},
                    ],
                }
            },
            "steps": [
                {
                    "target": "switch.doors_open_notification",
                    "action": SERVICE_TURN_ON,
                },
                {
                    "target": "switch.medicine_notification",
                    "action": SERVICE_TURN_ON,
                },
                {
                    "target": "switch.doors_open_notification",
                    "action": SERVICE_TURN_OFF,
                },
            ],
        },
    ),
)
async def test_toggle_notifications_with_shared_switches(
    hass: HomeAssistant,
    steps: list,
    configs: dict[str, dict[str, Any]],
    device_registry: dr.DeviceRegistry,
    entity_registry: er.EntityRegistry,
    snapshot: SnapshotAssertion,
) -> None:
    """Test turning on and off a notification."""
    for switch_id in {"light.entryway", "light.kitchen"} | {
        switch_id
        for config in configs.values()
        for switch_id in config.get(CONF_SWITCH_ENTITIES, [])
    }:
        add_mock_switch(hass, switch_id)

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
            **configs.get("doors_open", {}),
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
                "light.kitchen": ["medicine", "doors_open"],
            },
            **configs.get("medicine", {}),
        },
    )
    medicine_entry.add_to_hass(hass)
    await setup_integration(hass, medicine_entry)

    cluster_commands = async_mock_service(hass, "zha", "issue_zigbee_cluster_command")

    for step in steps:
        _LOGGER.log(TRACE, "step %r", step)

        await hass.services.async_call(
            SWITCH_DOMAIN,
            step["action"],
            {ATTR_ENTITY_ID: step["target"]},
            blocking=True,
        )

    assert cluster_commands == snapshot(name="zha_cluster_commands")

    for entity in [
        *entity_registry.entities.get_entries_for_config_entry_id(
            doors_open_entry.entry_id
        ),
        *entity_registry.entities.get_entries_for_config_entry_id(
            medicine_entry.entry_id
        ),
    ]:
        assert hass.states.get(entity.entity_id) == snapshot(name=entity.entity_id)


@Scenario.parametrize(
    Scenario(
        "color_override",
        {
            "configs": {
                "doors_open": {
                    CONF_START_ACTION: "script.color_override",
                }
            },
            "scripts": _response_script(
                "color_override", {"leds": [{"color": "cyan"}]}
            ),
            "steps": [
                {"target": "switch.doors_open_notification", "action": SERVICE_TURN_ON}
            ],
        },
    ),
    Scenario(
        "single_led_change",
        {
            "configs": {
                "doors_open": {
                    CONF_START_ACTION: "script.color_from_input",
                },
                "medicine": {
                    CONF_START_ACTION: "script.color_from_input",
                },
            },
            "scripts": {
                "color_from_input": {
                    "sequence": [
                        {
                            "variables": {
                                "lampie_response": (
                                    "{{ {'leds': "
                                    "[{'color': 'green' if notification == 'doors_open' else 'yellow'}] + "
                                    "[{}] * 6 "
                                    "} }}"
                                )
                            },
                        },
                        {
                            "stop": "done",
                            "response_variable": "lampie_response",
                        },
                    ]
                },
            },
            "steps": [
                {
                    "target": "switch.windows_open_notification",
                    "action": SERVICE_TURN_ON,
                },
                {"target": "switch.doors_open_notification", "action": SERVICE_TURN_ON},
            ],
        },
    ),
    Scenario(
        "block_activation",
        {
            "configs": {
                "doors_open": {
                    CONF_START_ACTION: "script.block_activation",
                }
            },
            "scripts": _response_script("block_activation", {"block_activation": True}),
            "steps": [
                {"target": "switch.doors_open_notification", "action": SERVICE_TURN_ON}
            ],
        },
    ),
    Scenario(
        "block_next",
        {
            "configs": {
                "doors_open": {
                    CONF_END_ACTION: "script.block_next",
                }
            },
            "scripts": _response_script("block_next", {"block_next": True}),
            "steps": [
                {
                    "target": "switch.windows_open_notification",
                    "action": SERVICE_TURN_ON,
                },
                {"target": "switch.doors_open_notification", "action": SERVICE_TURN_ON},
                {
                    "target": "switch.doors_open_notification",
                    "action": SERVICE_TURN_OFF,
                },
            ],
        },
    ),
)
async def test_toggle_notification_with_actions(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    switch: er.RegistryEntry,
    configs: dict[str, dict[str, Any]],
    scripts: dict[str, Any],
    steps: list,
    device_registry: dr.DeviceRegistry,
    entity_registry: er.EntityRegistry,
    snapshot: SnapshotAssertion,
    caplog,
) -> None:
    """Test turning on and off a notification."""

    await setup_integration(hass, config_entry)
    hass.config_entries.async_update_entry(
        config_entry,
        data={
            **config_entry.data,
            CONF_PRIORITY: {
                "light.kitchen": ["doors_open", "windows_open"],
            },
            **configs.get("doors_open", {}),
        },
    )

    windows_open_entry = MockConfigEntry(
        domain=DOMAIN,
        title="Windows Open",
        entry_id="mock-windows-open-id",
        data={
            CONF_COLOR: "orange",
            CONF_EFFECT: "slow_blink",
            CONF_SWITCH_ENTITIES: ["light.kitchen"],
            CONF_PRIORITY: {
                "light.kitchen": ["doors_open", "windows_open"],
            },
            **configs.get("medicine", {}),
        },
    )
    windows_open_entry.add_to_hass(hass)
    await setup_integration(hass, windows_open_entry)

    assert await async_setup_component(hass, "script", {"script": scripts})

    cluster_commands = async_mock_service(hass, "zha", "issue_zigbee_cluster_command")
    async_call = hass.services.async_call

    with patch(
        "homeassistant.core.ServiceRegistry.async_call", side_effect=async_call
    ) as mocked_service_call:
        for step in steps:
            _LOGGER.log(TRACE, "step %r", step)

            await async_call(
                SWITCH_DOMAIN,
                step["action"],
                {ATTR_ENTITY_ID: step["target"]},
                blocking=True,
            )

        service_calls = [
            (domain, service, args, *rest)
            for call in mocked_service_call.mock_calls
            for domain, service, args, *rest in [call.args]
            if (domain, service) != ("zha", "issue_zigbee_cluster_command")
        ]

    assert cluster_commands == snapshot(name="zha_cluster_commands")
    assert service_calls == snapshot(
        name="service_calls", matcher=any_device_id_matcher
    )

    for entity in entity_registry.entities.get_entries_for_config_entry_id(
        config_entry.entry_id
    ):
        assert hass.states.get(entity.entity_id) == snapshot(name=entity.entity_id)


@Scenario.parametrize(
    Scenario(
        "duration_expired_all",
        {
            "initial_leds_on": [True],
            "event": {"command": "led_effect_complete_ALL_LEDS"},
            "expected_notification_state": "off",
            "expected_leds_on": [],
            "expected_zha_calls": 0,
        },
    ),
    Scenario(
        "duration_expired_all_block_dismissal_ignored",
        {
            "configs": {
                "doors_open": {
                    CONF_END_ACTION: "script.block_dismissal",
                },
            },
            "scripts": _response_script("block_dismissal", {"block_dismissal": True}),
            "initial_leds_on": [True],
            "event": {"command": "led_effect_complete_ALL_LEDS"},
            "expected_notification_state": "off",
            "expected_leds_on": [],
            "expected_zha_calls": 0,
        },
    ),
    Scenario(
        "duration_expired_all_with_2x_tap_disabled",
        {
            "initial_leds_on": [True],
            "initial_states": {
                "switch.kitchen_disable_config_2x_tap_to_clear_notifications": "on",
            },
            "event": {"command": "led_effect_complete_ALL_LEDS"},
            "expected_notification_state": "off",
            "expected_leds_on": [],
            "expected_zha_calls": 0,
        },
    ),
    Scenario(
        "duration_expired_all_individual",
        {
            "initial_leds_on": [True] * 7,
            "event": {"command": "led_effect_complete_ALL_LEDS"},
            "expected_notification_state": "off",
            "expected_leds_on": [],
            "expected_zha_calls": 0,
        },
    ),
    Scenario(
        "duration_1_expired",
        {
            "initial_leds_on": [True] * 7,
            "event": {"command": "led_effect_complete_LED_1"},
            "expected_notification_state": "on",
            "expected_leds_on": [False] + [True] * 6,
            "expected_zha_calls": 0,
        },
    ),
    Scenario(
        "duration_7_expired_with_invalid_setup",
        {
            "initial_leds_on": [True],
            "event": {"command": "led_effect_complete_LED_7"},
            "expected_notification_state": "on",
            "expected_leds_on": [True],
            "expected_zha_calls": 0,
            "expected_log_messages": (
                "could not clear switch config at index 6 on light.kitchen"
            ),
        },
    ),
    *[
        scenario
        for prefix, command in [
            ("config", "button_3_double"),
            ("aux", "button_6_double"),
        ]
        for scenario in [
            Scenario(
                f"{prefix}_double_press",
                {
                    "initial_leds_on": [True],
                    "event": {"command": command},
                    "expected_notification_state": "off",
                    "expected_leds_on": [],
                    "expected_zha_calls": 0,
                },
            ),
            Scenario(
                f"{prefix}_double_press_with_block_dismissal_script",
                {
                    "configs": {
                        "doors_open": {
                            CONF_END_ACTION: "script.block_dismissal",
                        },
                    },
                    "scripts": _response_script(
                        "block_dismissal", {"block_dismissal": True}
                    ),
                    "initial_leds_on": [True],
                    "event": {"command": command},
                    "expected_notification_state": "on",
                    "expected_leds_on": [True],
                    "expected_zha_calls": 1,
                },
            ),
            Scenario(
                f"{prefix}_double_press_with_block_next_script",
                {
                    "configs": {
                        "doors_open": {
                            CONF_END_ACTION: "script.block_next",
                        },
                    },
                    "scripts": _response_script("block_next", {"block_next": True}),
                    "initial_leds_on": [True],
                    "event": {"command": command},
                    "expected_notification_state": "off",
                    "expected_leds_on": [],
                    "expected_zha_calls": 0,
                },
            ),
            Scenario(
                f"{prefix}_double_press_individual_leds_configured",
                {
                    "initial_leds_on": [True] * 6 + [False],
                    "event": {"command": command},
                    "expected_notification_state": "off",
                    "expected_leds_on": [],
                    "expected_zha_calls": 0,
                },
            ),
            Scenario(
                f"{prefix}_double_press_with_service_override",
                {
                    "initial_leds_on": [True],
                    "initial_led_config_source": LEDConfigSource(
                        f"{DOMAIN}.{SERVICE_NAME_OVERRIDE}", LEDConfigSourceType.SERVICE
                    ),
                    "event": {"command": command},
                    "expected_notification_state": "on",
                    "expected_leds_on": [True],
                    "expected_led_config_source": LEDConfigSource(
                        "doors_open", LEDConfigSourceType.NOTIFICATION
                    ),
                    "expected_zha_calls": 0,
                },
            ),
            Scenario(
                f"{prefix}_double_press_no_notification",
                {
                    "initial_leds_on": [True],
                    "initial_led_config_source": None,
                    "event": {"command": command},
                    "expected_notification_state": "on",
                    "expected_leds_on": [True],
                    "expected_led_config_source": None,
                    "expected_zha_calls": 0,
                    "expected_log_messages": (
                        "missing LED config and/or source for dismissal on switch light.kitchen; "
                        f"skipping processing ZHA event {command}"
                    ),
                },
            ),
            Scenario(
                f"{prefix}_double_with_local_protection_and_2x_tap_dismisses",
                {
                    "initial_leds_on": [True],
                    "initial_states": {
                        "switch.kitchen_local_protection": "on",
                    },
                    "event": {"command": command},
                    "expected_notification_state": "off",
                    "expected_leds_on": [],
                    "expected_zha_calls": 1,
                },
            ),
            Scenario(
                f"{prefix}_double_with_local_protection_and_2x_tap_and_block_dismissal_script",
                {
                    "configs": {
                        "doors_open": {
                            CONF_END_ACTION: "script.block_dismissal",
                        },
                    },
                    "scripts": _response_script(
                        "block_dismissal", {"block_dismissal": True}
                    ),
                    "initial_leds_on": [True],
                    "initial_states": {
                        "switch.kitchen_local_protection": "on",
                    },
                    "event": {"command": command},
                    "expected_notification_state": "on",
                    "expected_leds_on": [True],
                    "expected_zha_calls": 0,
                },
            ),
            Scenario(
                f"{prefix}_double_with_local_protection_and_2x_tap_disabled",
                {
                    "initial_leds_on": [True],
                    "initial_states": {
                        "switch.kitchen_local_protection": "on",
                        "switch.kitchen_disable_config_2x_tap_to_clear_notifications": "on",
                    },
                    "event": {"command": command},
                    "expected_notification_state": "on",
                    "expected_leds_on": [True],
                    "expected_zha_calls": 0,
                },
            ),
        ]
    ],
)
async def test_dismissal_from_switch(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    switch: er.RegistryEntry,
    configs: dict[str, dict[str, Any]],
    scripts: dict[str, Any],
    initial_leds_on: list[bool],
    initial_led_config_source: LEDConfigSource | None,
    initial_states: dict[str, str],
    event: dict[str, Any],
    expected_notification_state: str,
    expected_leds_on: list[bool],
    expected_led_config_source: LEDConfigSource | None,
    expected_zha_calls: int,
    expected_log_messages: str,
    device_registry: dr.DeviceRegistry,
    entity_registry: er.EntityRegistry,
    snapshot: SnapshotAssertion,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test turning on and off a notification."""

    def expand_led_config(flags):
        return tuple(
            LEDConfig(Color.BLUE, effect=Effect.SOLID if flag else Effect.CLEAR)
            for flag in flags
        )

    def expand_config_source(source, default_value):
        if source == Scenario.ABSENT:
            source = LEDConfigSource(default_value)
        return source

    await setup_integration(hass, config_entry)

    for entity_id, state in (initial_states or {}).items():
        hass.states.async_set(entity_id, state)

    initial_led_config = expand_led_config(initial_leds_on)
    initial_led_config_source = expand_config_source(
        initial_led_config_source, "doors_open"
    )
    expected_led_config = expand_led_config(expected_leds_on)
    expected_led_config_source = expand_config_source(
        expected_led_config_source,
        "doors_open" if expected_notification_state == "on" else None,
    )

    hass.config_entries.async_update_entry(
        config_entry,
        data={
            **config_entry.data,
            **(configs or {}).get("doors_open", {}),
            CONF_LED_CONFIG: [item.to_dict() for item in initial_led_config],
        },
    )
    await hass.async_block_till_done()

    orchestrator = config_entry.runtime_data.orchestrator
    orchestrator.store_notification_info("doors_open", notification_on=True)
    orchestrator.store_switch_info(
        switch.entity_id,
        led_config_source=initial_led_config_source,
        led_config=initial_led_config,
    )

    assert await async_setup_component(hass, "script", {"script": scripts})

    cluster_commands = async_mock_service(hass, "zha", "issue_zigbee_cluster_command")

    with patch(
        "homeassistant.core.ServiceRegistry.async_call",
        side_effect=hass.services.async_call,
    ) as mocked_service_call:
        hass.bus.async_fire(
            "zha_event",
            {
                "device_id": switch.device_id,
                **event,
            },
        )
        await hass.async_block_till_done()

        service_calls = [
            (domain, service, args, *rest)
            for call in mocked_service_call.mock_calls
            for domain, service, args, *rest in [call.args]
            if (domain, service) != ("zha", "issue_zigbee_cluster_command")
        ]

    assert (
        hass.states.get("switch.doors_open_notification").state
        == expected_notification_state
    )

    switch_info = orchestrator.switch_info(switch.entity_id)
    assert switch_info.led_config == expected_led_config
    assert switch_info.led_config_source == expected_led_config_source

    assert len(cluster_commands) == expected_zha_calls

    if expected_zha_calls:
        assert cluster_commands == snapshot(name="zha_cluster_commands")

    assert service_calls == snapshot(
        name="service_calls", matcher=any_device_id_matcher
    )

    if expected_log_messages:
        assert expected_log_messages in caplog.text


async def test_config_entries_linked_to_switch_device(
    hass: HomeAssistant,
    device_registry: dr.DeviceRegistry,
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
            CONF_SWITCH_ENTITIES: ["light.entryway"],
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
        },
    )
    medicine_entry.add_to_hass(hass)
    await setup_integration(hass, medicine_entry)
    medicine_device = device_registry.async_get_device(
        {(DOMAIN, medicine_entry.entry_id)}
    )

    device_ids = {
        device.id
        for device in device_registry.devices.values()
        if device.config_entries & {"mock-doors-open-id", "mock-medicine-id"}
    }

    assert device_ids == {
        doors_open_device.id,
        medicine_device.id,
        entryway_switch.device_id,
        kitchen_switch.device_id,
    }


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
