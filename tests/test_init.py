"""Test component setup."""

from collections import defaultdict
from collections.abc import Awaitable, Callable
import datetime as dt
import functools
import inspect
import logging
import re
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
from homeassistant.util import slugify
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_mock_service,
)
from syrupy.assertion import SnapshotAssertion
from syrupy.filters import props

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
from custom_components.lampie.orchestrator import LampieOrchestrator
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

from . import (
    _ANY,
    ANY,
    AbsentNone,
    MockNow,
    Scenario,
    add_mock_switch,
    setup_added_integration,
    setup_integration,
)
from .syrupy import any_device_id_matcher

_LOGGER = logging.getLogger(__name__)

type ANYType = _ANY
type ScenarioStageKwargs = dict[str, Any]
type ScenarioStageCallback = Callable[[], Awaitable[dict[str, Any] | None]]
type ScenarioStageHandler = Callable[
    [
        ScenarioStageCallback,
        ScenarioStageCallback,
        ScenarioStageCallback,
    ],
    Awaitable[None],
]


@pytest.fixture(name="configs")
def mock_configs() -> dict[str, Any]:
    return {}


@pytest.fixture(name="initial_states")
def mock_initial_states() -> dict[str, str]:
    return {}


@pytest.fixture(name="scripts")
def mock_scripts() -> dict[str, Any]:
    return {}


@pytest.fixture(name="expected_states")
def mock_expected_states() -> dict[str, str]:
    return {}


@pytest.fixture(name="expected_timers")
def mock_expected_timers() -> dict[str, bool]:
    return {}


@pytest.fixture(name="expected_events")
def mock_expected_events() -> int | ANYType | None:
    return ANY


@pytest.fixture(name="expected_service_calls")
def mock_expected_service_calls() -> int | ANYType | None:
    return ANY


@pytest.fixture(name="expected_zha_calls")
def mock_expected_zha_calls() -> int | ANYType | None:
    return ANY


@pytest.fixture(name="expected_log_messages")
def mock_expected_log_messages() -> str | None:
    return None


@pytest.fixture(name="snapshots")
def mock_snapshots() -> dict[str, Any]:
    return {}


def scenario_stages(
    fn: ScenarioStageHandler | None = None, *, standard_config_entry: bool | None = None
):
    if fn is not None:
        return _scenario_stages(fn, standard_config_entry=bool(standard_config_entry))
    return functools.partial(
        scenario_stages, standard_config_entry=standard_config_entry
    )


def _scenario_stages(
    fn: ScenarioStageHandler,
    *,
    standard_config_entry: bool = False,
):
    async def _standard_config_entry_signature(
        config_entry: MockConfigEntry,
        switch: er.RegistryEntry,
    ):
        pass

    @functools.wraps(fn)
    async def wrapper(**kwargs: Any) -> dict[str, Any]:
        extra_kwargs: dict[str, Any] = {}

        async def setup():
            extra_kwargs.update(result := await _setup(**(kwargs | extra_kwargs)))
            return result

        async def steps_callback():
            extra_kwargs.update(result := await _steps(**(kwargs | extra_kwargs)))
            return result

        async def assert_expectations():
            extra_kwargs.update(
                result := await _assert_expectations(**(kwargs | extra_kwargs))
            )
            return result

        await fn(setup, steps_callback, assert_expectations, **kwargs)

        return kwargs | extra_kwargs

    fn_fixture_parameters = []
    fn_callback_parameters_to_remove = 3

    for param in inspect.signature(fn).parameters.values():
        if fn_callback_parameters_to_remove and param.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            fn_callback_parameters_to_remove -= 1
            continue

        fn_fixture_parameters.append(param)

    wrapper_signature = inspect.Signature()
    wrapper_parameters = [
        *wrapper_signature.parameters.values(),
        *inspect.signature(_setup).parameters.values(),
        *inspect.signature(_steps).parameters.values(),
        *inspect.signature(_assert_expectations).parameters.values(),
        *fn_fixture_parameters,
    ]

    if standard_config_entry:
        wrapper_parameters += inspect.signature(
            _standard_config_entry_signature
        ).parameters.values()

    # remove duplicate param names
    wrapper_parameters = [
        *{(param.name): param for param in wrapper_parameters}.values()
    ]

    wrapper_signature = wrapper_signature.replace(
        parameters=[
            param
            for param in sorted(wrapper_parameters, key=lambda param: param.kind)
            if not param.name.startswith("_")
            and param.kind
            not in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }
        ],
    )
    wrapper.__signature__ = wrapper_signature  # type: ignore[attr-defined]

    return wrapper


async def _setup(
    hass: HomeAssistant,
    configs: dict[str, dict[str, Any]],
    initial_states: dict[str, str],
    scripts: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Scenario stage helper function for setup.

    Args:
        configs: Mapping of config entry slug to config entry data.
        initial_states: Mapping of entity ID to state value.
        scripts: A set of scripts to setup with the scripts integration.

        hass: Injected (do not configure in scenarios).
        kwargs: Catchall for additional injected values.
    """
    configs = configs or {}  # ensure AbsentNone is converted to dict
    initial_states = initial_states or {}
    standard_config_entry: MockConfigEntry | None = kwargs.get("config_entry")

    if standard_config_entry is not None:
        standard_config_entry.add_to_hass(hass)
        standard_config_data = dict((configs or {}).get("doors_open", {}))

        hass.config_entries.async_update_entry(
            standard_config_entry,
            data={
                **standard_config_entry.data,
                **standard_config_data,
            },
        )

        await setup_added_integration(hass, standard_config_entry)

    # register the any additional switches needed from config entries.
    for switch_id in {
        switch_id
        for config in configs.values()
        for switch_id in config.get(CONF_SWITCH_ENTITIES, [])
    }:
        add_mock_switch(hass, switch_id)

    # register the any additional switches needed from `initial_states`.
    for entity_id in initial_states:
        if entity_id.startswith("light."):
            add_mock_switch(hass, entity_id)

    # setup initial states
    for entity_id, state in initial_states.items():
        hass.states.async_set(entity_id, state)

    # add config entries based on those in the config items.
    for config_key, config_data in configs.items():
        if config_key == "doors_open" and standard_config_entry is not None:
            continue

        title = " ".join([part.capitalize() for part in config_key.split("_")])
        config_data = {**config_data}
        entry = MockConfigEntry(
            domain=DOMAIN,
            title=title,
            entry_id=f"mock-{config_key}:id",
            data=config_data,
        )
        entry.add_to_hass(hass)
        await setup_integration(hass, entry)

    if scripts is not None:
        assert await async_setup_component(hass, "script", {"script": scripts})

    cluster_commands = async_mock_service(hass, "zha", "issue_zigbee_cluster_command")

    return {
        "_cluster_commands": cluster_commands,
    }


async def _steps(
    hass: HomeAssistant,
    now: MockNow,
    steps: list[dict[str, Any]],
    entity_registry: er.EntityRegistry,
    **kwargs: Any,
) -> dict[str, Any]:
    """Scenario stage helper function for setup.

    Args:
        scripts: A set of scripts to setup with the scripts integration.
        steps: A list of mappings.

        hass: Injected (do not configure in scenarios).
        now: Injected (do not configure in scenarios).
        entity_registry: Injected (do not configure in scenarios).
        kwargs: Catchall for additional injected values.
    """
    standard_switch: er.RegistryEntry | None = kwargs.get("switch")
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

            # actions and events may be expanded by other config settings
            step_actions = []
            step_events = []

            if "action" in step:
                step_actions.append(step)
            elif "event" in step:
                step_events.append(step)
            elif "delay" in step:
                now._tick(step["delay"].total_seconds())

            if "device_config" in step:
                device_config = step["device_config"]
                target = device_config["target"]

                if "local_protection" in device_config:
                    local_protection_id = re.sub(
                        r"light\.(.*)", r"switch.\1_local_protection", target
                    )
                    hass.states.async_set(
                        local_protection_id, device_config["local_protection"]
                    )

                if "disable_clear_notification" in device_config:
                    disable_clear_notification_id = re.sub(
                        r"light\.(.*)",
                        r"switch.\1_disable_config_2x_tap_to_clear_notifications",
                        target,
                    )
                    hass.states.async_set(
                        disable_clear_notification_id,
                        device_config["disable_clear_notification"],
                    )

            for step in step_actions:
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

            for step in step_events:
                event_data = {**step["event"]}
                entity_id = event_data.pop("entity_id", None)
                device_id = (
                    standard_switch.device_id if standard_switch is not None else None
                )

                # if supplied, get the device related to the entity_id instead
                if entity_id is not None:
                    device_id = entity_registry.async_get(entity_id).device_id

                hass.bus.async_fire(
                    "zha_event",
                    {
                        "device_id": device_id,
                        **event_data,
                    },
                )

            await hass.async_block_till_done()

        # capture service calls to assert about start/end action invocations
        service_calls = [
            (domain, service, args, *rest)
            for call in mocked_service_call.mock_calls
            for domain, service, args, *rest in [call.args]
            if (domain, service) != ("zha", "issue_zigbee_cluster_command")
        ]

    return {
        "_service_calls": service_calls,
        "_events": events,
    }


async def _assert_expectations(  # noqa: RUF029
    hass: HomeAssistant,
    configs: dict[str, dict[str, Any]],
    initial_states: dict[str, str],
    _cluster_commands: list[Any],
    _service_calls: list[Any],
    _events: list[Any],
    expected_states: dict[str, str],
    expected_timers: dict[str, bool],
    expected_events: int | ANYType | None,
    expected_service_calls: int | ANYType | None,
    expected_zha_calls: int | ANYType | None,
    expected_log_messages: str | AbsentNone | None,
    entity_registry: er.EntityRegistry,
    snapshot: SnapshotAssertion,
    snapshots: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
    **kwargs: Any,
) -> dict[str, Any]:
    """Scenario stage helper function for asserting expectations.

    Args:
        configs: See `_setup`.
        initial_states: See `_setup`.
        expected_states: Mapping of entity ID to state value.
        expected_timers: Mapping of `notification|slug` or `switch|entity_id` to
            boolean value indicating if there should be a timer setup in the
            orchestrator.
        expected_events: Number of expected events that Lampie emits (or ANY to
            simply snapshot).
        expected_service_calls: Number of expected service calls that Lampie
            makes (or ANY to simply snapshot). This will be filtered to ignore
            service calls that are for sending messages to the device cluster.
            So it's usually just asserting against the `script` invocations used
            for start/end actions.
        expected_zha_calls: Number of expected service calls into the device
            cluster (or ANY to simply snapshot).
        expected_log_messages: Message to check to ensure it's logged.
        snapshots: A mapping for what to snapshot with the following keys:
            configs (default=`True`):  A boolean (reused for all config entries)
                or a mapping from config entry slugs to one of `bool`,
                `"standard_info"` or `"entities"`. `"standard_info"` will
                snapshot the orchestrator info for the standard config entry's
                notification and switch. `"entities"` can be used to snapshot
                all of the entities the config entry creates. `True` will do
                both and `False` will do neither.
            switches (default=`True`): Whether to snapshot switches.
            events (default=`True`): Whether to snapshot events.
            service_calls (default=`True`): Whether to snapshot service calls.
            cluster_commands (default=`True`): Whether to snapshot cluster
                commands.

        _cluster_commands: Internal (do not configure in scenarios).
        _service_calls: Internal (do not configure in scenarios).
        _events: Internal (do not configure in scenarios).
        hass: Injected (do not configure in scenarios).
        entity_registry: Injected (do not configure in scenarios).
        snapshot: Injected (do not configure in scenarios).
        caplog: Injected (do not configure in scenarios).
        kwargs: Catchall for additional injected values.
    """
    configs = configs or {}  # ensure AbsentNone is converted to dict
    standard_config_entry: MockConfigEntry | None = kwargs.get("config_entry")
    standard_switch: er.RegistryEntry | None = kwargs.get("switch")
    orchestrator: LampieOrchestrator = hass.data[DOMAIN]

    snapshot_configs = snapshots.get("configs", True)
    snapshot_switches = snapshots.get("switches", True)
    snapshot_events = snapshots.get("events", True)
    snapshot_service_calls = snapshots.get("service_calls", True)
    snapshot_cluster_commands = snapshots.get("cluster_commands", True)

    if isinstance(snapshot_configs, bool):
        value = snapshot_configs
        snapshot_configs = defaultdict(lambda: value)

    if (
        standard_config_entry is not None
        and standard_switch is not None
        and snapshot_configs.get("doors_open", True) in {True, "standard_info"}
    ):
        notification_info = orchestrator.notification_info("doors_open")
        switch_info = orchestrator.switch_info(standard_switch.entity_id)

        assert notification_info == snapshot(name="notification_info")
        assert switch_info == snapshot(name="switch_info")

    if expected_events and snapshot_events:
        assert _events == snapshot(name="events")

    if expected_service_calls and snapshot_service_calls:
        assert _service_calls == snapshot(
            name="service_calls", matcher=any_device_id_matcher
        )

    if expected_zha_calls and snapshot_cluster_commands:
        assert _cluster_commands == snapshot(name="zha_cluster_commands")

    # assert switch info against internal state to avoid creating the switch
    # info: this allows some test cases to assert that the orchestrator
    # never tries to track anything about the switch.
    if snapshot_switches:
        for entity_id in initial_states or {}:
            if entity_id.startswith("light."):
                assert orchestrator._switches.get(entity_id) == snapshot(
                    name=f"swtich_info:{entity_id}"
                )

    for entity in (
        entity
        for entry in hass.config_entries.async_entries(DOMAIN)
        for entity in entity_registry.entities.get_entries_for_config_entry_id(
            entry.entry_id
        )
        if snapshot_configs.get(slugify(entry.title), True) in {True, "entities"}
    ):
        assert hass.states.get(entity.entity_id) == snapshot(name=entity.entity_id)

    # make assertions from scenario input
    for entity_id, state in expected_states.items():
        assert hass.states.get(entity_id).state == state, (
            f"expected state of {entity_id} to be {state}"
        )

    for timer_spec, expected_timer in expected_timers.items():
        timer_type, lookup = timer_spec.split("|", 1)

        if timer_type == "notification":
            expiration = orchestrator.notification_info(lookup).expiration
        elif timer_type == "switch":
            expiration = orchestrator.switch_info(lookup).expiration
        else:
            raise AssertionError(
                f"bad timer type {timer_type} -- should be notification or"
                "switc, i.e. `notification|doors_open` "
                "or `switch|light.kitchen`"
            )

        assert (
            bool(expiration.cancel_listener) == expected_timer
        ), f"expected timer for {timer_spec} {
            'to exist' if expected_timer else 'not to exist'
        }"

    if snapshot_events and expected_events not in {None, Scenario.ABSENT, ANY}:
        assert len(_events) == expected_events, (
            f"expected {expected_events} lampie events to have been emitted"
        )

    if snapshot_service_calls and expected_service_calls not in {
        None,
        Scenario.ABSENT,
        ANY,
    }:
        assert len(_service_calls) == expected_service_calls, (
            f"expected {expected_service_calls} services calls (usually script invocations)"
        )

    if expected_zha_calls not in {None, Scenario.ABSENT, ANY}:
        assert len(_cluster_commands) == expected_zha_calls, (
            f"expected {expected_zha_calls} cluster command service calls"
        )

    if expected_log_messages not in {None, Scenario.ABSENT}:
        assert expected_log_messages in caplog.text

    return {}


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
            # compat for HA DeviceRegistryEntrySnapshot <2025.8.0 and >=2025.8.0
            "suggested_area",
        ),
    )


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
            "expected_states": {"switch.doors_open_notification": "off"},
            "expected_timers": {
                "notification|doors_open": False,
                "switch|light.kitchen": False,
            },
            "expected_events": 1,
            "expected_zha_calls": 1,
        },
    ),
    Scenario(
        "doors_open_various_firmware_durations_expired",
        {
            "configs": {
                "doors_open": {
                    CONF_LED_CONFIG: [
                        {CONF_COLOR: "blue", CONF_DURATION: 30},
                        {CONF_COLOR: "blue", CONF_DURATION: 300},
                        {CONF_COLOR: "blue", CONF_DURATION: 7200},
                        {CONF_COLOR: "blue", CONF_DURATION: 30},
                        {CONF_COLOR: "blue", CONF_DURATION: 7200},
                        {CONF_COLOR: "blue", CONF_DURATION: 300},
                        {CONF_COLOR: "blue", CONF_DURATION: 30},
                    ]
                }
            },
            "steps": [
                {
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                    "target": "switch.doors_open_notification",
                },
                {"event": {"command": "led_effect_complete_LED_1"}},
                {"event": {"command": "led_effect_complete_LED_4"}},
                {"event": {"command": "led_effect_complete_LED_7"}},
                {"event": {"command": "led_effect_complete_LED_2"}},
                {"event": {"command": "led_effect_complete_LED_6"}},
                {"event": {"command": "led_effect_complete_LED_3"}},
                {"event": {"command": "led_effect_complete_LED_5"}},
            ],
            "expected_states": {"switch.doors_open_notification": "off"},
            "expected_timers": {
                "notification|doors_open": False,
                "switch|light.kitchen": False,
            },
            "expected_events": 1,
            "expected_zha_calls": 7,
        },
    ),
    Scenario(
        "doors_open_5m1s_duration_unexpired",
        {
            "configs": {
                "doors_open": {
                    CONF_DURATION: dt.timedelta(minutes=5, seconds=1).total_seconds()
                }
            },
            "steps": [
                {
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                    "target": "switch.doors_open_notification",
                },
            ],
            "expected_states": {"switch.doors_open_notification": "on"},
            "expected_timers": {
                "notification|doors_open": True,
                "switch|light.kitchen": False,
            },
            "expected_events": 0,
            "expected_zha_calls": 1,
        },
    ),
    Scenario(
        "doors_open_5m1s_duration_expired",
        {
            "configs": {
                "doors_open": {
                    CONF_DURATION: dt.timedelta(minutes=5, seconds=1).total_seconds()
                }
            },
            "steps": [
                {
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                    "target": "switch.doors_open_notification",
                },
                {"delay": dt.timedelta(minutes=5, seconds=1)},
            ],
            "expected_states": {"switch.doors_open_notification": "off"},
            "expected_timers": {
                "notification|doors_open": False,
                "switch|light.kitchen": False,
            },
            "expected_events": 1,
            "expected_zha_calls": 2,
        },
    ),
    Scenario(
        "doors_open_mixed_specific_durations_partially_expired",
        {
            "configs": {
                "doors_open": {
                    CONF_LED_CONFIG: [
                        {CONF_COLOR: "red"},
                        {CONF_COLOR: "orange", CONF_DURATION: "0:05:01"},
                        {CONF_COLOR: "white", CONF_DURATION: "0:10:01"},
                        {CONF_COLOR: "red"},
                        {CONF_COLOR: "orange", CONF_DURATION: "0:05:01"},
                        {CONF_COLOR: "white", CONF_DURATION: "0:10:01"},
                        {CONF_COLOR: 0},
                    ],
                }
            },
            "steps": [
                {
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                    "target": "switch.doors_open_notification",
                },
                {"delay": dt.timedelta(minutes=5, seconds=2)},
            ],
            "expected_states": {"switch.doors_open_notification": "on"},
            "expected_timers": {
                "notification|doors_open": True,
                "switch|light.kitchen": False,
            },
            "expected_events": 0,
            "expected_zha_calls": 9,
        },
    ),
    Scenario(
        "doors_open_mixed_specific_durations_expired",
        {
            "configs": {
                "doors_open": {
                    CONF_LED_CONFIG: [
                        {CONF_COLOR: "red", CONF_DURATION: "0:05:01"},
                        {CONF_COLOR: "orange", CONF_DURATION: "0:05:01"},
                        {CONF_COLOR: "white", CONF_DURATION: "0:10:01"},
                        {CONF_COLOR: "red", CONF_DURATION: "0:05:01"},
                        {CONF_COLOR: "orange", CONF_DURATION: "0:05:01"},
                        {CONF_COLOR: "white", CONF_DURATION: "0:10:01"},
                        {CONF_COLOR: 0, CONF_DURATION: "0:05:01"},
                    ],
                }
            },
            "steps": [
                {
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                    "target": "switch.doors_open_notification",
                },
                {"delay": dt.timedelta(minutes=5, seconds=2)},
                {"delay": dt.timedelta(minutes=5, seconds=1)},
            ],
            "expected_states": {"switch.doors_open_notification": "off"},
            "expected_timers": {
                "notification|doors_open": False,
                "switch|light.kitchen": False,
            },
            "expected_events": 1,
            "expected_zha_calls": 14,
        },
    ),
    Scenario(
        "doors_open_5m1s_dismissed",
        {
            "configs": {
                "doors_open": {
                    CONF_DURATION: dt.timedelta(minutes=5, seconds=1).total_seconds()
                }
            },
            "steps": [
                {
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                    "target": "switch.doors_open_notification",
                },
                {"event": {"command": "button_3_double"}},
            ],
            "expected_states": {"switch.doors_open_notification": "off"},
            "expected_timers": {
                "notification|doors_open": False,
                "switch|light.kitchen": False,
            },
            "expected_events": 1,
            "expected_zha_calls": 1,
        },
    ),
    Scenario(
        "doors_open_5m1s_turned_off",
        {
            "configs": {
                "doors_open": {
                    CONF_DURATION: dt.timedelta(minutes=5, seconds=1).total_seconds()
                }
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
            "expected_states": {"switch.doors_open_notification": "off"},
            "expected_timers": {
                "notification|doors_open": False,
                "switch|light.kitchen": False,
            },
            "expected_events": 0,
            "expected_zha_calls": 2,
        },
    ),
    Scenario(
        "doors_open_5m1s_reactivated_and_unexpired",
        {
            "configs": {
                "doors_open": {
                    CONF_DURATION: dt.timedelta(minutes=5, seconds=1).total_seconds()
                }
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
            "expected_states": {"switch.doors_open_notification": "on"},
            "expected_timers": {
                "notification|doors_open": True,
                "switch|light.kitchen": False,
            },
            "expected_events": 0,
            "expected_zha_calls": 1,
        },
    ),
    Scenario(
        "doors_open_5m1s_duration_expired_not_blocked_via_end_action",
        {
            "configs": {
                "doors_open": {
                    CONF_DURATION: dt.timedelta(minutes=5, seconds=1).total_seconds(),
                    CONF_END_ACTION: "script.block_dismissal",
                }
            },
            "scripts": _response_script("block_dismissal", {"block_dismissal": True}),
            "steps": [
                {
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                    "target": "switch.doors_open_notification",
                },
                {"delay": dt.timedelta(minutes=5, seconds=2)},
            ],
            "expected_states": {"switch.doors_open_notification": "off"},
            "expected_timers": {
                "notification|doors_open": False,
                "switch|light.kitchen": False,
            },
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
            "expected_states": {"switch.doors_open_notification": "off"},
            "expected_timers": {
                "notification|doors_open": False,
                "switch|light.kitchen": False,
            },
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
            "expected_states": {"switch.doors_open_notification": "off"},
            "expected_timers": {
                "notification|doors_open": False,
                "switch|light.kitchen": False,
            },
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
            "expected_states": {"switch.doors_open_notification": "off"},
            "expected_timers": {
                "notification|doors_open": False,
                "switch|light.kitchen": False,
            },
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
            "expected_states": {"switch.doors_open_notification": "off"},
            "expected_timers": {
                "notification|doors_open": False,
                "switch|light.kitchen": False,
            },
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
            "expected_states": {"switch.doors_open_notification": "off"},
            "expected_timers": {
                "notification|doors_open": False,
                "switch|light.kitchen": False,
            },
            "expected_events": 1,
            "expected_zha_calls": 1,
        },
    ),
    Scenario(
        "kitchen_override_leds_5m1s_duration_unexpired",
        {
            "configs": {},
            "steps": [
                {
                    "action": f"{DOMAIN}.{SERVICE_NAME_OVERRIDE}",
                    "target": "light.kitchen",
                    "data": {
                        ATTR_LED_CONFIG: [
                            {ATTR_COLOR: "green", ATTR_DURATION: "0:05:01"},
                        ],
                    },
                },
            ],
            "expected_states": {"switch.doors_open_notification": "off"},
            "expected_timers": {
                "notification|doors_open": False,
                "switch|light.kitchen": True,
            },
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
            "expected_states": {"switch.doors_open_notification": "on"},
            "expected_timers": {
                "notification|doors_open": False,
                "switch|light.kitchen": False,
            },
            "expected_events": 0,
            "expected_zha_calls": 1,
        },
    ),
    Scenario(
        "doors_open_on,kitchen_override_leds_reset",
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
                {
                    "action": f"{DOMAIN}.{SERVICE_NAME_OVERRIDE}",
                    "target": "light.kitchen",
                    "data": {
                        ATTR_LED_CONFIG: None,
                    },
                },
            ],
            "expected_states": {"switch.doors_open_notification": "on"},
            "expected_timers": {
                "notification|doors_open": False,
                "switch|light.kitchen": False,
            },
            "expected_events": 0,
            "expected_zha_calls": 3,
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
            "expected_states": {"switch.doors_open_notification": "on"},
            "expected_timers": {
                "notification|doors_open": False,
                "switch|light.kitchen": False,
            },
            "expected_events": 1,
            "expected_zha_calls": 2,
        },
    ),
    Scenario(
        "doors_open_on,kitchen_override_leds_5m1s_duration_expired",
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
                            {ATTR_COLOR: "green", ATTR_DURATION: "0:05:01"},
                        ],
                    },
                },
                {"delay": dt.timedelta(minutes=5, seconds=2)},
            ],
            "expected_states": {"switch.doors_open_notification": "on"},
            "expected_timers": {
                "notification|doors_open": False,
                "switch|light.kitchen": False,
            },
            "expected_events": 1,
            "expected_zha_calls": 3,
        },
    ),
    Scenario(
        "kitchen_override_leds_5m1s_duration_expired",
        {
            "configs": {},
            "steps": [
                {
                    "action": f"{DOMAIN}.{SERVICE_NAME_OVERRIDE}",
                    "target": "light.kitchen",
                    "data": {
                        ATTR_LED_CONFIG: [
                            {ATTR_COLOR: "green", ATTR_DURATION: "0:05:01"},
                        ],
                    },
                },
                {"delay": dt.timedelta(minutes=5, seconds=2)},
            ],
            "expected_states": {"switch.doors_open_notification": "off"},
            "expected_timers": {
                "notification|doors_open": False,
                "switch|light.kitchen": False,
            },
            "expected_events": 1,
            "expected_zha_calls": 2,
        },
    ),
    Scenario(
        "kitchen_override_leds_mixed_specific_durations_partially_expired",
        {
            "configs": {},
            "steps": [
                {
                    "action": f"{DOMAIN}.{SERVICE_NAME_OVERRIDE}",
                    "target": "light.kitchen",
                    "data": {
                        ATTR_LED_CONFIG: [
                            {CONF_COLOR: "red", CONF_DURATION: "0:10:01"},
                            {CONF_COLOR: "orange", CONF_DURATION: "0:05:01"},
                            {CONF_COLOR: "white", CONF_DURATION: "0:10:01"},
                            {CONF_COLOR: "red", CONF_DURATION: "0:10:01"},
                            {CONF_COLOR: "orange", CONF_DURATION: "0:05:01"},
                            {CONF_COLOR: "white", CONF_DURATION: "0:10:01"},
                            {CONF_COLOR: 0, CONF_DURATION: "0:10:01"},
                        ],
                    },
                },
                {"delay": dt.timedelta(minutes=5, seconds=2)},
            ],
            "expected_states": {"switch.doors_open_notification": "off"},
            "expected_timers": {
                "notification|doors_open": False,
                "switch|light.kitchen": True,
            },
            "expected_events": 0,
            "expected_zha_calls": 9,
        },
    ),
    Scenario(
        "kitchen_override_leds_mixed_specific_durations_expired",
        {
            "configs": {},
            "steps": [
                {
                    "action": f"{DOMAIN}.{SERVICE_NAME_OVERRIDE}",
                    "target": "light.kitchen",
                    "data": {
                        ATTR_LED_CONFIG: [
                            {CONF_COLOR: "red", CONF_DURATION: "0:05:01"},
                            {CONF_COLOR: "orange", CONF_DURATION: "0:05:01"},
                            {CONF_COLOR: "white", CONF_DURATION: "0:10:01"},
                            {CONF_COLOR: "red", CONF_DURATION: "0:05:01"},
                            {CONF_COLOR: "orange", CONF_DURATION: "0:05:01"},
                            {CONF_COLOR: "white", CONF_DURATION: "0:10:01"},
                            {CONF_COLOR: 0, CONF_DURATION: "0:05:01"},
                        ],
                    },
                },
                {"delay": dt.timedelta(minutes=5, seconds=2)},
                {"delay": dt.timedelta(minutes=5, seconds=1)},
            ],
            "expected_states": {"switch.doors_open_notification": "off"},
            "expected_timers": {
                "notification|doors_open": False,
                "switch|light.kitchen": False,
            },
            "expected_events": 1,
            "expected_zha_calls": 14,
        },
    ),
    Scenario(
        "kitchen_override_leds_5m1s_duration_dismissed",
        {
            "configs": {},
            "steps": [
                {
                    "action": f"{DOMAIN}.{SERVICE_NAME_OVERRIDE}",
                    "target": "light.kitchen",
                    "data": {
                        ATTR_LED_CONFIG: [
                            {ATTR_COLOR: "green", ATTR_DURATION: "0:05:01"},
                        ],
                    },
                },
                {"event": {"command": "button_3_double"}},
            ],
            "expected_states": {"switch.doors_open_notification": "off"},
            "expected_timers": {
                "notification|doors_open": False,
                "switch|light.kitchen": False,
            },
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
            "expected_states": {"switch.doors_open_notification": "off"},
            "expected_timers": {
                "notification|doors_open": False,
                "switch|light.kitchen": False,
            },
            "expected_events": 0,
            "expected_zha_calls": 1,
        },
    ),
    Scenario(
        "entryway_override_expired",
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
                {
                    "event": {
                        "command": "led_effect_complete_ALL_LEDS",
                        "entity_id": "light.entryway",
                    },
                },
            ],
            "expected_states": {"switch.doors_open_notification": "off"},
            "expected_timers": {
                "notification|doors_open": False,
                "switch|light.kitchen": False,
            },
            "expected_events": 1,
            "expected_zha_calls": 1,
        },
    ),
    Scenario(
        "entryway_no_override_dismissed",
        {
            "configs": {},
            "initial_states": {
                "light.entryway": "on",
            },
            "steps": [
                {
                    "event": {
                        "command": "button_3_double",
                        "entity_id": "light.entryway",
                    },
                },
            ],
            "expected_states": {"switch.doors_open_notification": "off"},
            "expected_timers": {
                "notification|doors_open": False,
                "switch|light.kitchen": False,
            },
            "expected_events": 0,
            "expected_zha_calls": 0,
        },
    ),
)
@scenario_stages(standard_config_entry=True)
async def test_toggle_notifications(
    setup: ScenarioStageCallback,
    steps_callback: ScenarioStageCallback,
    assert_expectations: ScenarioStageCallback,
    **kwargs: Any,
):
    await setup()
    await steps_callback()
    await assert_expectations()


_TOGGLE_NOTIFICATIONS_WITH_SHARED_SWITCHES_DOORS_OPEN = {
    CONF_COLOR: "red",
    CONF_EFFECT: "open_close",
    CONF_SWITCH_ENTITIES: ["light.entryway", "light.kitchen"],
    CONF_PRIORITY: {
        "light.kitchen": ["medicine", "doors_open"],
    },
}

_TOGGLE_NOTIFICATIONS_WITH_SHARED_SWITCHES_MEDICINE = {
    CONF_COLOR: "cyan",
    CONF_EFFECT: "slow_blink",
    CONF_SWITCH_ENTITIES: ["light.kitchen"],
    CONF_PRIORITY: {
        "light.kitchen": ["medicine", "doors_open"],
    },
}

_TOGGLE_NOTIFICATIONS_WITH_SHARED_SWITCHES_CONFIGS = {
    "doors_open": _TOGGLE_NOTIFICATIONS_WITH_SHARED_SWITCHES_DOORS_OPEN,
    "medicine": _TOGGLE_NOTIFICATIONS_WITH_SHARED_SWITCHES_MEDICINE,
}


@Scenario.parametrize(
    Scenario(
        "doors_open_on",
        {
            "configs": _TOGGLE_NOTIFICATIONS_WITH_SHARED_SWITCHES_CONFIGS,
            "steps": [
                {
                    "target": "switch.doors_open_notification",
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                }
            ],
            "expected_events": None,  # disabled for all (via none/absent)
            "expected_service_calls": None,  # disabled for all (via none/absent)
        },
    ),
    Scenario(
        "medicine_on",
        {
            "configs": _TOGGLE_NOTIFICATIONS_WITH_SHARED_SWITCHES_CONFIGS,
            "steps": [
                {
                    "target": "switch.medicine_notification",
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                }
            ],
        },
    ),
    Scenario(
        "both_on",
        {
            "configs": _TOGGLE_NOTIFICATIONS_WITH_SHARED_SWITCHES_CONFIGS,
            "steps": [
                {
                    "target": "switch.doors_open_notification",
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                },
                {
                    "target": "switch.medicine_notification",
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                },
            ],
        },
    ),
    Scenario(
        "doors_open_on_with_third_switch",
        {
            "configs": _TOGGLE_NOTIFICATIONS_WITH_SHARED_SWITCHES_CONFIGS
            | {
                "doors_open": {
                    **_TOGGLE_NOTIFICATIONS_WITH_SHARED_SWITCHES_DOORS_OPEN,
                    CONF_SWITCH_ENTITIES: ["light.entryway", "light.dining_room"],
                    CONF_PRIORITY: {
                        "light.dining_room": ["medicine", "doors_open"],
                    },
                },
                "medicine": {
                    **_TOGGLE_NOTIFICATIONS_WITH_SHARED_SWITCHES_MEDICINE,
                    CONF_SWITCH_ENTITIES: ["light.dining_room", "light.kitchen"],
                    CONF_PRIORITY: {
                        "light.dining_room": ["medicine", "doors_open"],
                    },
                },
            },
            "steps": [
                {
                    "target": "switch.doors_open_notification",
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                },
            ],
        },
    ),
    Scenario(
        "both_on_then_medicine_off",
        {
            "configs": _TOGGLE_NOTIFICATIONS_WITH_SHARED_SWITCHES_CONFIGS,
            "steps": [
                {
                    "target": "switch.doors_open_notification",
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                },
                {
                    "target": "switch.medicine_notification",
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                },
                {
                    "target": "switch.medicine_notification",
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_OFF}",
                },
            ],
        },
    ),
    Scenario(
        "both_on_then_doors_open_off",
        {
            "configs": _TOGGLE_NOTIFICATIONS_WITH_SHARED_SWITCHES_CONFIGS,
            "steps": [
                {
                    "target": "switch.doors_open_notification",
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                },
                {
                    "target": "switch.medicine_notification",
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                },
                {
                    "target": "switch.doors_open_notification",
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_OFF}",
                },
            ],
        },
    ),
    Scenario(
        "both_on_then_doors_open_off_custom_medicine_leds",
        {
            "configs": _TOGGLE_NOTIFICATIONS_WITH_SHARED_SWITCHES_CONFIGS
            | {
                "medicine": {
                    **_TOGGLE_NOTIFICATIONS_WITH_SHARED_SWITCHES_MEDICINE,
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
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                },
                {
                    "target": "switch.medicine_notification",
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                },
                {
                    "target": "switch.doors_open_notification",
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_OFF}",
                },
            ],
        },
    ),
)
@scenario_stages
async def test_toggle_notifications_with_shared_switches(
    setup: ScenarioStageCallback,
    steps_callback: ScenarioStageCallback,
    assert_expectations: ScenarioStageCallback,
    **kwargs: Any,
):
    await setup()
    await steps_callback()
    await assert_expectations()


_TOGGLE_NOTIFICATION_WITH_ACTIONS_DOORS_OPEN = {
    CONF_PRIORITY: {
        "light.kitchen": ["doors_open", "windows_open"],
    },
}

_TOGGLE_NOTIFICATION_WITH_ACTIONS_WINDOWS_OPEN = {
    CONF_COLOR: "orange",
    CONF_EFFECT: "slow_blink",
    CONF_SWITCH_ENTITIES: ["light.kitchen"],
    CONF_PRIORITY: {
        "light.kitchen": ["doors_open", "windows_open"],
    },
}

_TOGGLE_NOTIFICATION_WITH_ACTIONS_CONFIGS = {
    "doors_open": _TOGGLE_NOTIFICATION_WITH_ACTIONS_DOORS_OPEN,
    "windows_open": _TOGGLE_NOTIFICATION_WITH_ACTIONS_WINDOWS_OPEN,
}

_TOGGLE_NOTIFICATION_WITH_ACTIONS_SNAPSHOTS = {
    "configs": {
        "doors_open": "entities",
        "windows_open": False,
    }
}

_TOGGLE_NOTIFICATION_WITH_ACTIONS_BASE = {
    "expected_events": None,
    "snapshots": _TOGGLE_NOTIFICATION_WITH_ACTIONS_SNAPSHOTS,
}


@Scenario.parametrize(
    Scenario(
        "color_override",
        {
            **_TOGGLE_NOTIFICATION_WITH_ACTIONS_BASE,
            "configs": {
                **_TOGGLE_NOTIFICATION_WITH_ACTIONS_CONFIGS,
                "doors_open": {
                    **_TOGGLE_NOTIFICATION_WITH_ACTIONS_DOORS_OPEN,
                    CONF_START_ACTION: "script.color_override",
                },
            },
            "scripts": _response_script(
                "color_override", {"leds": [{"color": "cyan"}]}
            ),
            "steps": [
                {
                    "target": "switch.doors_open_notification",
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                }
            ],
            "expected_events": None,  # disabled for all (via none/absent)
        },
    ),
    Scenario(
        "single_led_change",
        {
            **_TOGGLE_NOTIFICATION_WITH_ACTIONS_BASE,
            "configs": {
                **_TOGGLE_NOTIFICATION_WITH_ACTIONS_CONFIGS,
                "doors_open": {
                    **_TOGGLE_NOTIFICATION_WITH_ACTIONS_DOORS_OPEN,
                    CONF_START_ACTION: "script.color_from_input",
                },
                "windows_open": {
                    **_TOGGLE_NOTIFICATION_WITH_ACTIONS_WINDOWS_OPEN,
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
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                },
                {
                    "target": "switch.doors_open_notification",
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                },
            ],
        },
    ),
    Scenario(
        "block_activation",
        {
            **_TOGGLE_NOTIFICATION_WITH_ACTIONS_BASE,
            "configs": {
                **_TOGGLE_NOTIFICATION_WITH_ACTIONS_CONFIGS,
                "doors_open": {
                    **_TOGGLE_NOTIFICATION_WITH_ACTIONS_DOORS_OPEN,
                    CONF_START_ACTION: "script.block_activation",
                },
            },
            "scripts": _response_script("block_activation", {"block_activation": True}),
            "steps": [
                {
                    "target": "switch.doors_open_notification",
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                }
            ],
        },
    ),
    Scenario(
        "block_next",
        {
            **_TOGGLE_NOTIFICATION_WITH_ACTIONS_BASE,
            "configs": {
                **_TOGGLE_NOTIFICATION_WITH_ACTIONS_CONFIGS,
                "doors_open": {
                    **_TOGGLE_NOTIFICATION_WITH_ACTIONS_DOORS_OPEN,
                    CONF_END_ACTION: "script.block_next",
                },
            },
            "scripts": _response_script("block_next", {"block_next": True}),
            "steps": [
                {
                    "target": "switch.windows_open_notification",
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                },
                {
                    "target": "switch.doors_open_notification",
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_ON}",
                },
                {
                    "target": "switch.doors_open_notification",
                    "action": f"{SWITCH_DOMAIN}.{SERVICE_TURN_OFF}",
                },
            ],
        },
    ),
)
@scenario_stages(standard_config_entry=True)
async def test_toggle_notification_with_actions(
    setup: ScenarioStageCallback,
    steps_callback: ScenarioStageCallback,
    assert_expectations: ScenarioStageCallback,
    **kwargs: Any,
):
    await setup()
    await steps_callback()
    await assert_expectations()


_DISMISSAL_FROM_SWITCH_SNAPSHOTS = {
    "configs": {
        "doors_open": False,
    },
    "events": False,
}

_DISMISSAL_FROM_SWITCH_BASE = {
    "snapshots": _DISMISSAL_FROM_SWITCH_SNAPSHOTS,
}


@Scenario.parametrize(
    Scenario(
        "duration_expired_all",
        {
            **_DISMISSAL_FROM_SWITCH_BASE,
            "initial_leds_on": [True],
            "steps": [{"event": {"command": "led_effect_complete_ALL_LEDS"}}],
            "expected_notification_state": "off",
            "expected_leds_on": [],
            "expected_zha_calls": 0,
        },
    ),
    Scenario(
        "duration_expired_all_block_dismissal_ignored",
        {
            **_DISMISSAL_FROM_SWITCH_BASE,
            "configs": {
                "doors_open": {
                    CONF_END_ACTION: "script.block_dismissal",
                },
            },
            "scripts": _response_script("block_dismissal", {"block_dismissal": True}),
            "initial_leds_on": [True],
            "steps": [{"event": {"command": "led_effect_complete_ALL_LEDS"}}],
            "expected_notification_state": "off",
            "expected_leds_on": [],
            "expected_zha_calls": 0,
        },
    ),
    Scenario(
        "duration_expired_all_with_2x_tap_disabled",
        {
            **_DISMISSAL_FROM_SWITCH_BASE,
            "initial_leds_on": [True],
            "initial_states": {
                "switch.kitchen_disable_config_2x_tap_to_clear_notifications": "on",
            },
            "steps": [{"event": {"command": "led_effect_complete_ALL_LEDS"}}],
            "expected_notification_state": "off",
            "expected_leds_on": [],
            "expected_zha_calls": 0,
        },
    ),
    Scenario(
        "duration_expired_all_individual",
        {
            **_DISMISSAL_FROM_SWITCH_BASE,
            "initial_leds_on": [True] * 7,
            "steps": [{"event": {"command": "led_effect_complete_ALL_LEDS"}}],
            "expected_notification_state": "off",
            "expected_leds_on": [],
            "expected_zha_calls": 0,
        },
    ),
    Scenario(
        "duration_1_expired",
        {
            **_DISMISSAL_FROM_SWITCH_BASE,
            "initial_leds_on": [True] * 7,
            "steps": [{"event": {"command": "led_effect_complete_LED_1"}}],
            "expected_notification_state": "on",
            "expected_leds_on": [False] + [True] * 6,
            "expected_zha_calls": 0,
        },
    ),
    Scenario(
        "duration_7_expired_with_invalid_setup",
        {
            **_DISMISSAL_FROM_SWITCH_BASE,
            "initial_leds_on": [True],
            "steps": [{"event": {"command": "led_effect_complete_LED_7"}}],
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
                    **_DISMISSAL_FROM_SWITCH_BASE,
                    "initial_leds_on": [True],
                    "steps": [{"event": {"command": command}}],
                    "expected_notification_state": "off",
                    "expected_leds_on": [],
                    "expected_zha_calls": 0,
                },
            ),
            Scenario(
                f"{prefix}_double_press_with_block_dismissal_script",
                {
                    **_DISMISSAL_FROM_SWITCH_BASE,
                    "configs": {
                        "doors_open": {
                            CONF_END_ACTION: "script.block_dismissal",
                        },
                    },
                    "scripts": _response_script(
                        "block_dismissal", {"block_dismissal": True}
                    ),
                    "initial_leds_on": [True],
                    "steps": [{"event": {"command": command}}],
                    "expected_notification_state": "on",
                    "expected_leds_on": [True],
                    "expected_zha_calls": 1,
                },
            ),
            Scenario(
                f"{prefix}_double_press_with_block_next_script",
                {
                    **_DISMISSAL_FROM_SWITCH_BASE,
                    "configs": {
                        "doors_open": {
                            CONF_END_ACTION: "script.block_next",
                        },
                    },
                    "scripts": _response_script("block_next", {"block_next": True}),
                    "initial_leds_on": [True],
                    "steps": [{"event": {"command": command}}],
                    "expected_notification_state": "off",
                    "expected_leds_on": [],
                    "expected_zha_calls": 0,
                },
            ),
            Scenario(
                f"{prefix}_double_press_individual_leds_configured",
                {
                    **_DISMISSAL_FROM_SWITCH_BASE,
                    "initial_leds_on": [True] * 6 + [False],
                    "steps": [{"event": {"command": command}}],
                    "expected_notification_state": "off",
                    "expected_leds_on": [],
                    "expected_zha_calls": 0,
                },
            ),
            Scenario(
                f"{prefix}_double_press_with_service_override",
                {
                    **_DISMISSAL_FROM_SWITCH_BASE,
                    "initial_leds_on": [True],
                    "initial_led_config_source": LEDConfigSource(
                        f"{DOMAIN}.{SERVICE_NAME_OVERRIDE}", LEDConfigSourceType.SERVICE
                    ),
                    "steps": [{"event": {"command": command}}],
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
                    **_DISMISSAL_FROM_SWITCH_BASE,
                    "initial_leds_on": [True],
                    "initial_led_config_source": None,
                    "steps": [{"event": {"command": command}}],
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
                    **_DISMISSAL_FROM_SWITCH_BASE,
                    "initial_leds_on": [True],
                    "initial_states": {
                        "switch.kitchen_local_protection": "on",
                    },
                    "steps": [{"event": {"command": command}}],
                    "expected_notification_state": "off",
                    "expected_leds_on": [],
                    "expected_zha_calls": 1,
                },
            ),
            Scenario(
                f"{prefix}_double_with_local_protection_and_2x_tap_and_block_dismissal_script",
                {
                    **_DISMISSAL_FROM_SWITCH_BASE,
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
                    "steps": [{"event": {"command": command}}],
                    "expected_notification_state": "on",
                    "expected_leds_on": [True],
                    "expected_zha_calls": 0,
                },
            ),
            Scenario(
                f"{prefix}_double_with_local_protection_and_2x_tap_disabled",
                {
                    **_DISMISSAL_FROM_SWITCH_BASE,
                    "initial_leds_on": [True],
                    "initial_states": {
                        "switch.kitchen_local_protection": "on",
                        "switch.kitchen_disable_config_2x_tap_to_clear_notifications": "on",
                    },
                    "steps": [{"event": {"command": command}}],
                    "expected_notification_state": "on",
                    "expected_leds_on": [True],
                    "expected_zha_calls": 0,
                },
            ),
        ]
    ],
)
@scenario_stages(standard_config_entry=True)
async def test_dismissal_from_switch(
    setup: ScenarioStageCallback,
    steps_callback: ScenarioStageCallback,
    assert_expectations: ScenarioStageCallback,
    *,
    hass: HomeAssistant,
    configs: dict[str, dict[str, Any]],
    initial_leds_on: list[bool],
    initial_led_config_source: LEDConfigSource | None,
    expected_notification_state: str,
    expected_leds_on: list[bool],
    expected_led_config_source: LEDConfigSource | None,
    expected_log_messages: str,
    **kwargs: Any,
):
    configs = configs or {}  # ensure AbsentNone is converted to dict
    standard_config_entry: MockConfigEntry = kwargs["config_entry"]
    standard_switch: er.RegistryEntry = kwargs["switch"]

    def expand_led_config(flags):
        return tuple(
            LEDConfig(Color.BLUE, effect=Effect.SOLID if flag else Effect.CLEAR)
            for flag in flags
        )

    def expand_config_source(source, default_value):
        if source == Scenario.ABSENT:
            source = LEDConfigSource(default_value)
        return source

    # calculate actual initial LED config & source
    initial_led_config = expand_led_config(initial_leds_on)
    initial_led_config_source = expand_config_source(
        initial_led_config_source, "doors_open"
    )

    # calculate actual expected LED config & source
    expected_led_config = expand_led_config(expected_leds_on)
    expected_led_config_source = expand_config_source(
        expected_led_config_source,
        "doors_open" if expected_notification_state == "on" else None,
    )

    await setup()

    hass.config_entries.async_update_entry(
        standard_config_entry,
        data={
            **standard_config_entry.data,
            **configs.get("doors_open", {}),
            CONF_LED_CONFIG: [item.to_dict() for item in initial_led_config],
        },
    )
    await hass.async_block_till_done()

    orchestrator = standard_config_entry.runtime_data.orchestrator
    orchestrator.store_notification_info("doors_open", notification_on=True)
    orchestrator.store_switch_info(
        standard_switch.entity_id,
        led_config_source=initial_led_config_source,
        led_config=initial_led_config,
    )

    await steps_callback()
    await assert_expectations()

    assert (
        hass.states.get("switch.doors_open_notification").state
        == expected_notification_state
    )

    switch_info = orchestrator.switch_info(standard_switch.entity_id)
    assert switch_info.led_config == expected_led_config
    assert switch_info.led_config_source == expected_led_config_source


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
