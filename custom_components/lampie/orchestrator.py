"""The Lampie orchestrator."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import replace
import datetime as dt
from enum import Enum, IntEnum, auto
from functools import partial
import logging
import re
from typing import TYPE_CHECKING, Any, Final, NamedTuple, Protocol, Unpack

from homeassistant.components import mqtt
from homeassistant.components.mqtt import DOMAIN as MQTT_DOMAIN
from homeassistant.components.mqtt.models import MqttData
from homeassistant.components.script import DOMAIN as SCRIPT_DOMAIN
from homeassistant.components.switch import DOMAIN as SWITCH_DOMAIN
from homeassistant.core import CALLBACK_TYPE, Event, HomeAssistant, callback
from homeassistant.helpers import (
    device_registry as dr,
    entity_registry as er,
    event as evt,
)
from homeassistant.helpers.device import (
    async_device_info_to_link_from_entity,
    async_entity_id_to_device_id,
)
from homeassistant.helpers.json import json_dumps
from homeassistant.util import dt as dt_util
from homeassistant.util.hass_dict import HassKey
from homeassistant.util.json import json_loads_object

from .const import (
    CONF_END_ACTION,
    CONF_PRIORITY,
    CONF_START_ACTION,
    CONF_SWITCH_ENTITIES,
    TRACE,
)
from .types import (
    Color,
    DeviceId,
    Effect,
    ExpirationInfo,
    Integration,
    LampieNotificationInfo,
    LampieNotificationOptionsDict,
    LampieSwitchInfo,
    LampieSwitchOptionsDict,
    LEDConfig,
    LEDConfigSource,
    LEDConfigSourceType,
    Slug,
    SwitchId,
    Z2MSwitchInfo,
    ZHASwitchInfo,
)

if TYPE_CHECKING:
    from .coordinator import LampieUpdateCoordinator

type ZHAEventData = dict[str, Any]
type MQTTDeviceName = str

_LOGGER = logging.getLogger(__name__)

ZHA_DOMAIN: Final = "zha"
DATA_MQTT: HassKey[MqttData] = HassKey("mqtt")
ALREADY_EXPIRED: Final = 0

SWITCH_INTEGRATIONS = {
    ZHA_DOMAIN: Integration.ZHA,
    MQTT_DOMAIN: Integration.Z2M,
}

FIRMWARE_SECONDS_MAX = dt.timedelta(seconds=60).total_seconds()
FIRMWARE_MINUTES_MAX = dt.timedelta(minutes=60).total_seconds()
FIRMWARE_HOURS_MAX = dt.timedelta(hours=134).total_seconds()

CLEAR_CONFIG = LEDConfig(
    color=Color.BLUE,
    effect=Effect.CLEAR,
)

# the ZHA commands are used as the canonical command representation for this
# integration. other integrations map their command representation to these
# before using methods such as `_handle_generic_event`.
DISMISSAL_COMMANDS = {
    "button_3_double",
    "button_6_double",
    "led_effect_complete_ALL_LEDS",
    "led_effect_complete_LED_1",
    "led_effect_complete_LED_2",
    "led_effect_complete_LED_3",
    "led_effect_complete_LED_4",
    "led_effect_complete_LED_5",
    "led_effect_complete_LED_6",
    "led_effect_complete_LED_7",
}

CONFIGURABLE_COMMANDS = {
    "button_3_double",
    "button_6_double",
}

PHYSICAL_DISMISSAL_COMMANDS = {
    "button_3_double",
    "button_6_double",
}

Z2M_ACTION_TO_COMMAND_MAP = {
    "config_double": "button_3_double",
}


class _LEDMode(IntEnum):
    ALL = 1
    INDIVIDUAL = 3


class _DismissalBlocked(Exception):
    pass


class _SwitchKeyType(Enum):
    DEVICE_ID = auto()
    MQTT_NAME = auto()  # name from MQTT's discovery `state_topic`


class _SwitchKey[T](NamedTuple):
    type: _SwitchKeyType
    identifier: T


class _StartScriptResult(NamedTuple):
    led_config: tuple[LEDConfig, ...] | None
    block_activation: bool


class _EndScriptResult(NamedTuple):
    block_dismissal: bool
    block_next: bool


class _UnknownIntegrationError(Exception):
    pass


class _LampieUnmanagedSwitchCoordinator:
    def async_update_listeners(self) -> None:
        pass


if TYPE_CHECKING:

    class _ExpirationHandler(Protocol):
        def __call__(self, led_config: tuple[LEDConfig, ...]) -> Awaitable[None]: ...

    class _ExpirationHandlerDismissCall(Protocol):
        def __call__(self, dismissal_command: str) -> Awaitable[None]: ...

    class _ExpirationHandlerStoreExpirationCall(Protocol):
        def __call__(self, *, expiration: ExpirationInfo) -> None: ...


class LampieOrchestrator:
    """Lampie orchestrator."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize."""
        super().__init__()

        self._hass = hass
        self._coordinators: dict[Slug, LampieUpdateCoordinator] = {}
        self._notifications: dict[Slug, LampieNotificationInfo] = {}
        self._switches: dict[SwitchId, LampieSwitchInfo] = {}
        self._switch_ids: dict[_SwitchKey[DeviceId | MQTTDeviceName], SwitchId] = {}
        self._mqtt_subscriptions: set[str] = set()
        self._teardown_callbacks: list[CALLBACK_TYPE] = [
            hass.bus.async_listen(
                "zha_event",
                self._handle_zha_event,
                self._filter_zha_events,
            )
        ]

    async def add_coordinator(self, coordinator: LampieUpdateCoordinator) -> None:
        self._coordinators[coordinator.slug] = coordinator
        await self._update_references()

    async def remove_coordinator(self, coordinator: LampieUpdateCoordinator) -> None:
        self._coordinators.pop(coordinator.slug)
        await self._update_references()

    async def setup(self) -> None:
        pass

    def teardown(self) -> bool:
        if len(self._coordinators) == 0:
            for callback in self._teardown_callbacks:
                callback()

            for key, expiration in (
                *((key, info.expiration) for key, info in self._notifications.items()),
                *((key, info.expiration) for key, info in self._switches.items()),
            ):
                self._cancel_expiration(expiration, log_context=f"teardown:{key}")
            return True
        return False

    def switch_info(self, switch_id: SwitchId) -> LampieSwitchInfo:
        return self._switches[switch_id]

    def store_switch_info(
        self, switch_id: SwitchId, **kwargs: Unpack[LampieSwitchOptionsDict]
    ) -> None:
        """Update and store switch info.

        The primary Lampie coordinator responsible for the switch is notified,
        so entities will be notified of the change and updated.
        """
        assert len(kwargs.keys() & {"led_config", "led_config_source"}) in {0, 2}, (
            "led_config and led_config_source must be provided together"
        )

        self._switches[switch_id] = replace(
            self.switch_info(switch_id),
            **kwargs,
        )
        self._primary_for_switch(switch_id).async_update_listeners()

        if "led_config" in kwargs:
            source: LEDConfigSource = kwargs["led_config_source"]
            config: tuple[LEDConfig, ...] = kwargs["led_config"]

            _LOGGER.debug(
                "switch updated (%s=%s:%s)",
                switch_id,
                source,
                "off" if _all_clear(config) else "on",
            )

    async def _ensure_switch_setup_completed(self, switch_id: SwitchId) -> None:
        if switch_id in self._switches:
            return

        entity_registry = er.async_get(self._hass)
        device_registry = dr.async_get(self._hass)
        device_id = async_entity_id_to_device_id(self._hass, switch_id)
        device = device_registry.async_get(device_id)
        integration = self._switch_integration(device) if device else None
        entity_entries = er.async_entries_for_device(entity_registry, device_id)

        if not integration:
            raise _UnknownIntegrationError

        _LOGGER.debug(
            "searching %s related entries: %s",
            switch_id,
            [
                (
                    entry.domain,
                    entry.platform,
                    entry.unique_id,
                    entry.translation_key,
                )
                for entry in entity_entries
            ],
        )

        integration_info = await {
            Integration.ZHA: self._zha_switch_setup,
            Integration.Z2M: self._z2m_switch_setup,
        }[integration](switch_id, device, entity_entries)

        self._switch_ids[_SwitchKey(_SwitchKeyType.DEVICE_ID, device_id)] = switch_id
        self._switches[switch_id] = LampieSwitchInfo(
            integration=integration,
            integration_info=integration_info,
            led_config=(),
            led_config_source=LEDConfigSource(None),
        )

    async def _zha_switch_setup(  # noqa: PLR6301
        self,
        switch_id: SwitchId,  # noqa: ARG002
        device: dr.DeviceEntry,  # noqa: ARG002
        entity_entries: list[er.RegistryEntry],
    ) -> ZHASwitchInfo:
        local_protection_id = None
        disable_clear_notification_id = None

        def is_zha_platform(entry: er.RegistryEntry) -> bool:
            return bool(entry.platform == ZHA_DOMAIN)

        for entity_entry in filter(is_zha_platform, entity_entries):
            if (
                entity_entry.domain == SWITCH_DOMAIN
                and entity_entry.translation_key == "local_protection"
            ):
                local_protection_id = entity_entry.entity_id
                _LOGGER.debug(
                    "found ZHA local protection entity: %s",
                    local_protection_id,
                )
            if entity_entry.domain == SWITCH_DOMAIN and (
                entity_entry.translation_key == "disable_clear_notifications_double_tap"
            ):
                disable_clear_notification_id = entity_entry.entity_id
                _LOGGER.debug(
                    "found ZHA disable clear notification entity: %s",
                    disable_clear_notification_id,
                )

        return ZHASwitchInfo(
            local_protection_id=local_protection_id,
            disable_clear_notification_id=disable_clear_notification_id,
        )

    async def _z2m_switch_setup(
        self,
        switch_id: SwitchId,
        device: dr.DeviceEntry,
        entity_entries: list[er.RegistryEntry],  # noqa: ARG002
    ) -> Z2MSwitchInfo | None:
        if not await mqtt.async_wait_for_mqtt_client(self._hass):
            _LOGGER.error("MQTT integration is not available")
            return None

        try:
            full_topic = self._hass.data[DATA_MQTT].debug_info_entities[switch_id][
                "discovery_data"
            ]["discovery_payload"]["state_topic"]

            _LOGGER.debug(
                "found Z2M full topic for switch %s: %s", switch_id, full_topic
            )
        except (KeyError, AttributeError):
            full_topic = f"zigbee2mqtt/{device.name}"

            _LOGGER.warning(
                "failed to determine MQTT topic from internal HASS state for %s. "
                "using default of %s from device name",
                switch_id,
                full_topic,
            )

        # pull out the base topic and the device name from the full topic.
        base_topic, mqtt_device_name = full_topic.rsplit("/", 1)

        self._switch_ids[_SwitchKey(_SwitchKeyType.MQTT_NAME, mqtt_device_name)] = (
            switch_id
        )

        # subscribe to the base topic for state updates of `localProtection` and
        # `doubleTapClearNotifications` as well as the `action` updates for
        # interaction on switches.
        if base_topic not in self._mqtt_subscriptions:
            _LOGGER.debug("subscribing to %s for %s", base_topic, switch_id)
            self._mqtt_subscriptions.add(base_topic)
            self._teardown_callbacks.append(
                await mqtt.async_subscribe(
                    self._hass,
                    f"{base_topic}/+",
                    self._handle_z2m_message,
                )
            )

        # request an update to attributes we need to know the value of later.
        # these will be processed in the event handler.
        await self._hass.services.async_call(
            MQTT_DOMAIN,
            "publish",
            {
                "topic": f"{full_topic}/get",
                "payload": json_dumps(
                    {
                        "localProtection": "",
                        "doubleTapClearNotifications": "",
                    }
                ),
            },
            blocking=True,
        )

        return Z2MSwitchInfo(full_topic=full_topic)

    def has_notification(self, slug: Slug) -> bool:
        return slug in self._coordinators

    def notification_info(self, slug: Slug) -> LampieNotificationInfo:
        if slug not in self._notifications:
            self._notifications[slug] = LampieNotificationInfo()
        return self._notifications[slug]

    def store_notification_info(
        self,
        slug: Slug,
        **kwargs: Unpack[LampieNotificationOptionsDict],
    ) -> None:
        """Update and store notification info.

        The Lampie coordinator responsible for the notification is notified, so
        so entities will be notified of the change and updated.
        """
        self._notifications[slug] = replace(
            self.notification_info(slug),
            **kwargs,
        )
        self._coordinators[slug].async_update_listeners()

        if "notification_on" in kwargs:
            _LOGGER.debug(
                "notification updated (%s=%s)",
                slug,
                "on" if kwargs["notification_on"] else "off",
            )

    async def activate_notification(
        self,
        slug: Slug,
        *,
        led_config_source: LEDConfigSource | None = None,
        led_config: tuple[LEDConfig, ...] | None = None,
    ) -> None:
        """Activate a notification.

        Perform all necessary actions and validations to activate a
        notification, and update all switches as necessary based on the outcome.
        Switches may be updated to display this notification or left displaying
        a higher priority notification.
        """
        coordinator = self._coordinators[slug]
        entry = coordinator.config_entry
        switches = entry.data[CONF_SWITCH_ENTITIES]

        start_action_response = await self._invoke_start_action(
            slug,
            led_config
            or self.notification_info(slug).led_config_override
            or coordinator.led_config,
        )
        block_activation = start_action_response.block_activation

        if block_activation:
            _LOGGER.debug("notification activation blocked (%s)", slug)
            return

        self.store_notification_info(
            slug,
            notification_on=True,
            expiration=self._schedule_fallback_expiration(
                self.notification_info(slug).expiration,
                led_config
                or start_action_response.led_config
                or coordinator.led_config,
                partial(self._async_handle_notification_expired, slug),
                has_expiration_messaging=self._any_has_expiration_messaging(switches),
                log_context=slug,
            ),
        )

        # loop while waiting sequentially to avoid flooding the network
        for switch_id in switches:
            await self._switch_apply_notification_or_override(
                switch_id,
                led_config_source=led_config_source,
                led_config=led_config,
                assert_can_activate=True,
                log_context=f"activate-notification {slug}",
            )

    async def dismiss_notification(
        self,
        slug: Slug,
        *,
        dismissal_command: str | None = None,
        via_switch: SwitchId | None = None,
    ) -> None:
        """Dismiss a notification.

        Perform all necessary actions and validations to dismiss a notification,
        and update all switches as necessary based on the outcome. Switches may
        be updated to display another notification or cleared.

        Raises:
            _DismissalBlocked: Raised for internal event processing to handle
                dismissals that get blocked.
        """
        coordinator = self._coordinators[slug]
        entry = coordinator.config_entry
        switches = entry.data[CONF_SWITCH_ENTITIES]
        dismissed = dismissal_command in PHYSICAL_DISMISSAL_COMMANDS

        end_action_response = await self._invoke_end_action(
            slug,
            switch_id=via_switch,
            dismissed=dismissed,
        )
        block_dismissal = end_action_response.block_dismissal
        block_next = end_action_response.block_next

        if dismissed and block_dismissal:
            raise _DismissalBlocked

        self.store_notification_info(
            slug,
            notification_on=False,
            led_config_override=None,
            expiration=self._cancel_expiration(
                self.notification_info(slug).expiration,
                log_context=slug,
            ),
        )

        # loop while waiting sequentially to avoid flooding the network
        for switch_id in switches:
            await self._switch_apply_notification_or_override(
                switch_id,
                exclude={LEDConfigSourceType.NOTIFICATION} if block_next else None,
                log_context="dismiss-notification",
            )

        if dismissal_command:
            self._hass.bus.async_fire(
                "lampie.dismissed" if dismissed else "lampie.expired",
                {
                    "notification": slug,
                },
            )

    async def override_switch(
        self,
        switch_id: SwitchId,
        *,
        led_config_source: LEDConfigSource,
        led_config: tuple[LEDConfig, ...] | None,
    ) -> None:
        """Entrypoint for services to override & update a switch.

        The switch state will be stored and the physical switch will be updated
        to show the specified configuration.
        """

        # since this may be a switch that's not part of any of the config
        # entries, we need to start by ensuring it's set up.
        await self._ensure_switch_setup_completed(switch_id)

        from_state = self.switch_info(switch_id)
        is_reset = led_config is None

        self.store_switch_info(
            switch_id,
            expiration=self._schedule_fallback_expiration(
                from_state.expiration,
                led_config or (),
                partial(self._async_handle_switch_override_expired, switch_id),
                has_expiration_messaging=self._any_has_expiration_messaging(
                    [switch_id]
                ),
                log_context=f"{led_config_source}",
            ),
        )

        await self._switch_apply_notification_or_override(
            switch_id,
            led_config_source=led_config_source,
            led_config=led_config,
            exclude={LEDConfigSourceType.OVERRIDE} if is_reset else set(),
            log_context="override-switch",
        )

    async def _switch_apply_notification_or_override(
        self,
        switch_id: SwitchId,
        *,
        led_config_source: LEDConfigSource | None = None,
        led_config: tuple[LEDConfig, ...] | None = None,
        exclude: set[LEDConfigSourceType] | None = None,
        dismissal_command: str | None = None,
        strip_durations_for_firmware: bool = False,
        assert_can_activate: bool = False,
        log_context: str,
    ) -> None:
        """Update a switch and the state of the system.

        Args:
            switch_id: The switch to target.
            led_config: The pre-defined LED config to use rather than
                determining the value from the current state of the system. Must
                also provide `led_config_source`.
            led_config_source: The source of the `led_config`.
            exclude: The source types to exclude from presentation.
            dismissal_command: The ZHA or internal command that triggered
                the dismissal. This is used to determine if this was a physical
                dismissal that correlates to emitting an event for end users.
            strip_durations_for_firmware: Remove the durations before they're
                sent to the device. This should be used if the orchestrator is
                managing durations/expirations instead of the device.
            assert_can_activate: Adds a protective assertion in the method
                ensuring that there is a notification that can be activated.
                Useful for initial presentation of notifications.
            log_context: A short string to include in log messages.
        """
        from_state = self.switch_info(switch_id)
        activate = self._first_on_for_switch(switch_id)
        exclude = set() if exclude is None else exclude
        dismissed = dismissal_command in PHYSICAL_DISMISSAL_COMMANDS

        if assert_can_activate:
            assert activate is not None, (
                f"notification lookup failed for {switch_id}; from: {log_context}, "
                f"priorities: {self._switches[switch_id].priorities}"
            )

        if LEDConfigSourceType.NOTIFICATION in exclude:
            activate = None

        if led_config is None:
            led_config_source = LEDConfigSource(activate.slug if activate else None)

            if (
                from_state.led_config_source
                and from_state.led_config_source.type == LEDConfigSourceType.OVERRIDE
                and LEDConfigSourceType.OVERRIDE not in exclude
            ):
                led_config_source = from_state.led_config_source
                led_config = from_state.led_config
            elif activate:
                led_config = (
                    self.notification_info(activate.slug).led_config_override
                    or activate.led_config
                )
            else:
                led_config = ()
        else:
            assert led_config_source is not None, (
                "must provide led_config_source when providing led_config"
            )

        self.store_switch_info(
            switch_id, led_config_source=led_config_source, led_config=led_config
        )

        # skip sending durations to switch firmware if requested
        if strip_durations_for_firmware:
            _LOGGER.log(TRACE, "removing durations before sending to hardware")
            led_config = tuple(replace(led, duration=None) for led in led_config)

        _LOGGER.log(TRACE, "will transition switch: %s", switch_id)
        await self._transition_switch(
            switch_id,
            from_config=from_state.led_config,
            to_config=led_config,
        )

        if dismissal_command:
            self._hass.bus.async_fire(
                "lampie.dismissed" if dismissed else "lampie.expired",
                {
                    "entity_id": switch_id,
                    "override": from_state.led_config_source
                    and from_state.led_config_source.value,
                },
            )

    def _first_for_switch(
        self,
        switch_id: SwitchId,
        matcher: Callable[[LampieUpdateCoordinator], bool],
    ) -> LampieUpdateCoordinator | None:
        for slug in self._switches[switch_id].priorities:
            if (coordinator := self._coordinators[slug]) and matcher(coordinator):
                return coordinator
        return None

    def _first_on_for_switch(
        self,
        switch_id: SwitchId,
    ) -> LampieUpdateCoordinator | None:
        return self._first_for_switch(
            switch_id,
            lambda coordinator: self.notification_info(
                coordinator.slug
            ).notification_on,
        )

    def _primary_for_switch(
        self, switch_id: SwitchId
    ) -> LampieUpdateCoordinator | _LampieUnmanagedSwitchCoordinator:
        priorities = self._switches[switch_id].priorities
        coordinator = self._coordinators.get(priorities[0]) if priorities else None
        return coordinator or _LampieUnmanagedSwitchCoordinator()

    async def _invoke_start_action(
        self, slug: Slug, led_config: tuple[LEDConfig, ...]
    ) -> _StartScriptResult:
        coordinator = self._coordinators[slug]
        entry = coordinator.config_entry
        start_action = entry.data.get(CONF_START_ACTION)
        response = {}

        if start_action:
            args = {
                "notification": slug,
                "leds": [item.to_dict() for item in led_config],
            }

            _LOGGER.debug(
                "executing start action %r for notification %r (%r)",
                start_action,
                slug,
                args,
            )
            response = await self._hass.services.async_call(
                SCRIPT_DOMAIN,
                start_action.split(".")[1],
                args,
                blocking=True,
                return_response=True,
            )
            _LOGGER.debug("start action response: %r", response)

            if "leds" in response:
                led_config = tuple(
                    LEDConfig.from_config(item) for item in response["leds"]
                )
                self.store_notification_info(
                    slug,
                    led_config_override=led_config,
                )
                response = dict(response, leds=led_config)

        return _StartScriptResult(
            led_config=response.get("leds"),
            block_activation=response.get("block_activation", False),
        )

    async def _invoke_end_action(
        self, slug: Slug, *, switch_id: SwitchId | None, dismissed: bool
    ) -> _EndScriptResult:
        coordinator = self._coordinators[slug]
        entry = coordinator.config_entry
        end_action = entry.data.get(CONF_END_ACTION)
        response = {}

        if end_action:
            args = {
                "notification": slug,
                "switch_id": None,
                "device_id": None,
                "dismissed": False,
                **(
                    {
                        "switch_id": switch_id,
                        "device_id": async_entity_id_to_device_id(
                            self._hass, switch_id
                        ),
                        "dismissed": dismissed,
                    }
                    if switch_id
                    else {}
                ),
            }
            _LOGGER.debug(
                "executing end action %r for notification %r (%r)",
                end_action,
                slug,
                args,
            )
            response = await self._hass.services.async_call(
                SCRIPT_DOMAIN,
                end_action.split(".")[1],
                args,
                blocking=True,
                return_response=True,
            )
            _LOGGER.debug("end action response: %r", response)

        return _EndScriptResult(
            block_next=response.get("block_next", False),
            block_dismissal=response.get("block_dismissal", False),
        )

    async def _transition_switch(
        self,
        switch_id: SwitchId,
        *,
        from_config: tuple[LEDConfig, ...],
        to_config: tuple[LEDConfig, ...],
    ) -> None:
        _LOGGER.log(
            TRACE, "transition_switch: %s; %s -> %s", switch_id, from_config, to_config
        )

        from_mode = (
            None
            if not from_config
            else _LEDMode.ALL
            if len(from_config) == 1
            else _LEDMode.INDIVIDUAL
        )
        to_mode = _LEDMode.ALL if len(to_config) <= 1 else _LEDMode.INDIVIDUAL

        # clear the prior notification if the mode changed because `ALL` takes
        # priority over `INDIVIDUAL` on the switches. no changes need to be
        # applied if the changes were via the switch firmware since the switch
        # will already have cleared the notification.
        if from_mode and (from_mode != to_mode or (not to_config)):
            clear_config = {
                _LEDMode.ALL: (CLEAR_CONFIG,),
                _LEDMode.INDIVIDUAL: (CLEAR_CONFIG,) * 7,
            }[from_mode]

            _LOGGER.log(
                TRACE, "changing mode: %s; %s -> %s", switch_id, from_mode, to_mode
            )

            await self._issue_switch_commands(
                switch_id, clear_config, from_mode, from_config=from_config
            )

        if from_mode != to_mode:
            from_config = ()

        await self._issue_switch_commands(
            switch_id, to_config, to_mode, from_config=from_config
        )

    async def _issue_switch_commands(
        self,
        switch_id: SwitchId,
        led_config: tuple[LEDConfig, ...],
        led_mode: _LEDMode,
        *,
        from_config: tuple[LEDConfig, ...],
    ) -> None:
        _LOGGER.log(
            TRACE, "issue_switch_commands: %s; %s, %s", switch_id, led_mode, led_config
        )

        from_params = [
            self._switch_command_led_params(led, switch_id) for led in from_config
        ]
        device_info = async_device_info_to_link_from_entity(self._hass, switch_id)
        updated_leds = []

        # actually apply the desired changes either for the full LED bar or for
        # individual lights within the bar. only apply changes when needed to
        # reduce the number of messages sent and how much the bars/lights flash.
        # no changes need to be applied if the changes were via the switch
        # firmware since the switch will already have cleared the notification.
        for idx, led in enumerate(led_config):
            params = self._switch_command_led_params(led, switch_id)

            with suppress(IndexError):
                if params == from_params[idx]:
                    _LOGGER.log(TRACE, "skipping update of LED %s; no change", idx)
                    continue

            if led_mode == _LEDMode.INDIVIDUAL:
                params["led_number"] = idx
                updated_leds.append(str(idx))

            _LOGGER.log(TRACE, "update LED %s command: %r", idx, params)
            await self._dispatch_service_command(
                switch_id=switch_id,
                device_info=device_info,
                led_mode=led_mode,
                params=params,
            )

        _LOGGER.debug(
            "issued commands to update %s leds: %s",
            switch_id,
            "all" if led_mode == _LEDMode.ALL else ", ".join(updated_leds),
        )

    async def _dispatch_service_command(
        self,
        *,
        switch_id: SwitchId,
        device_info: dr.DeviceInfo,
        led_mode: _LEDMode,
        params: dict[str, Any],
    ) -> None:
        switch_info = self.switch_info(switch_id)
        service_command = {
            Integration.ZHA: self._zha_service_command,
            Integration.Z2M: self._z2m_service_command,
        }[switch_info.integration]

        await service_command(
            switch_info=switch_info,
            device_info=device_info,
            led_mode=led_mode,
            params=params,
        )

    async def _zha_service_command(
        self,
        *,
        switch_info: LampieSwitchInfo,  # noqa: ARG002
        device_info: dr.DeviceInfo,
        led_mode: _LEDMode,
        params: dict[str, Any],
    ) -> None:
        id_tuple = next(iter(device_info["identifiers"]))
        ieee = id_tuple[1]

        _LOGGER.log(TRACE, "zha.issue_zigbee_cluster_command: %s", led_mode)
        await self._hass.services.async_call(
            ZHA_DOMAIN,
            "issue_zigbee_cluster_command",
            {
                "ieee": ieee,
                "endpoint_id": 1,
                "cluster_id": 64561,
                "cluster_type": "in",
                "command": int(led_mode),
                "command_type": "server",
                "params": params,
                "manufacturer": 4655,
            },
            blocking=True,
        )

    async def _z2m_service_command(
        self,
        *,
        switch_info: LampieSwitchInfo,
        device_info: dr.DeviceInfo,  # noqa: ARG002
        led_mode: _LEDMode,
        params: dict[str, Any],
    ) -> None:
        command = (
            "individual_led_effect" if led_mode == _LEDMode.INDIVIDUAL else "led_effect"
        )
        zha_info = switch_info.integration_info
        topic = f"{zha_info.full_topic}/set"

        _LOGGER.log(TRACE, "mqtt.publish: %s", topic)
        await self._hass.services.async_call(
            MQTT_DOMAIN,
            "publish",
            {
                "topic": topic,
                "payload": json_dumps({command: params}),
            },
            blocking=True,
        )

    def _switch_command_led_params(
        self, led: LEDConfig, switch_id: SwitchId
    ) -> dict[str, Any]:
        firmware_duration = (
            self._firmware_duration(led.duration)
            if self._any_has_expiration_messaging([switch_id])
            else None
        )

        return {
            "led_color": int(led.color),
            "led_effect": led.effect.value,
            "led_level": int(led.brightness),
            "led_duration": 255
            if led.duration is None
            or led.duration == ALREADY_EXPIRED
            or firmware_duration is None
            else firmware_duration,
        }

    def _local_protection_enabled(self, switch_id: SwitchId) -> bool:
        switch_info = self.switch_info(switch_id)
        integration = switch_info.integration
        integration_info = switch_info.integration_info

        assert integration_info is not None, (
            f"missing integration info for switch {switch_id}"
        )

        def _zha() -> bool:
            zha_info: ZHASwitchInfo = integration_info

            return bool(
                local_protection_id := zha_info.local_protection_id
            ) and self._hass.states.is_state(local_protection_id, "on")

        def _z2m() -> bool:
            z2m_info: Z2MSwitchInfo = integration_info
            return bool(z2m_info.local_protection_enabled)

        return {
            Integration.ZHA: _zha,
            Integration.Z2M: _z2m,
        }[integration]()

    def _double_tap_clear_notifications_disabled(self, switch_id: SwitchId) -> bool:
        switch_info = self.switch_info(switch_id)
        integration = switch_info.integration
        integration_info = switch_info.integration_info

        assert integration_info is not None, (
            f"missing integration info for switch {switch_id}"
        )

        def _zha() -> bool:
            zha_info: ZHASwitchInfo = integration_info

            return bool(
                disable_clear_notification_id := zha_info.disable_clear_notification_id
            ) and self._hass.states.is_state(disable_clear_notification_id, "on")

        def _z2m() -> bool:
            z2m_info: Z2MSwitchInfo = integration_info
            return bool(z2m_info.double_tap_clear_notifications_disabled)

        return {
            Integration.ZHA: _zha,
            Integration.Z2M: _z2m,
        }[integration]()

    def _any_has_expiration_messaging(self, switches: Sequence[SwitchId]) -> bool:
        """Check if any switch via an integration that sends expiration events.

        All Inovelli Blue series switches send Zigbee messages for notifications
        expiring (when local protection is not enabled). However, integrations
        currently handle them differently:
            ZHA: Processes these and turns them into
                `led_effect_complete_{ALL|LED_1-7}` commands.
            Z2M: Processes these and turns them into
                `{"notificationComplete": "{ALL|LED_1-7}"}` messages.

        Returns:
            A boolean indicating if any of the switches are part of such an
                integration.
        """
        return any(
            self.switch_info(switch_id).integration
            in {Integration.ZHA, Integration.Z2M}
            for switch_id in switches
        )

    @staticmethod
    def _firmware_duration(seconds: int | None) -> int | None:
        """Convert a timeframe to a duration supported by the switch firmware.

        Args:
            seconds: The duration as a number of seconds.

        Returns:
            The duration parameter value (0-255) if it can be handled by the
                firmware or None if it cannot be.
        """
        if seconds is None or seconds == ALREADY_EXPIRED:
            return None
        if seconds <= FIRMWARE_SECONDS_MAX:
            return seconds
        if seconds <= FIRMWARE_MINUTES_MAX and seconds % 60 == 0:
            return 60 + seconds // 60
        if seconds <= FIRMWARE_HOURS_MAX and seconds % 3600 == 0:
            return 120 + seconds // 3600
        return None

    @callback
    def _filter_zha_events(
        self,
        event_data: ZHAEventData,
    ) -> bool:
        switch_key = _SwitchKey(_SwitchKeyType.DEVICE_ID, event_data["device_id"])
        return (
            switch_key in self._switch_ids
            and event_data["command"] in DISMISSAL_COMMANDS
        )

    @callback
    async def _handle_zha_event(self, event: Event[ZHAEventData]) -> None:
        device_id = event.data["device_id"]
        switch_key = _SwitchKey(_SwitchKeyType.DEVICE_ID, device_id)
        switch_id = self._switch_ids[switch_key]
        await self._handle_generic_event(
            command=event.data["command"],
            switch_id=switch_id,
            integration=Integration.ZHA,
        )

    @callback
    async def _handle_z2m_message(self, message: mqtt.ReceiveMessage) -> None:
        _LOGGER.log(TRACE, "handling mqtt message: %s", message)

        _, mqtt_device_name = message.topic.rsplit("/", 1)
        switch_key = _SwitchKey(_SwitchKeyType.MQTT_NAME, mqtt_device_name)
        switch_id = self._switch_ids.get(switch_key)

        # ignore the message if it's not related to a switch in the system
        if switch_id is None:
            return

        switch_info = self.switch_info(switch_id)
        zha_info = switch_info.integration_info
        zha_info_changed = False

        # ignore the message if for some reason there's a topic mismatch, i.e.
        # two devices named with different base topics.
        if message.topic != zha_info.full_topic:
            return

        # wait to parse the payload until the above checks are performed to avoid
        # the overhead of parsing messages we won't handle.
        payload = json_loads_object(message.payload)

        # if this is for a valid action, dispatch it to the generic event handler
        if (action := payload.get("action")) and (
            command := Z2M_ACTION_TO_COMMAND_MAP.get(action)
        ):
            await self._handle_generic_event(
                command=command,
                switch_id=switch_id,
                integration=Integration.Z2M,
            )

        if (
            (notification_complete := payload.get("notificationComplete"))
            and (command := f"led_effect_complete_{notification_complete}")
            and command in DISMISSAL_COMMANDS
        ):
            await self._handle_generic_event(
                command=command,
                switch_id=switch_id,
                integration=Integration.Z2M,
            )

        if "localProtection" in payload:
            zha_info_changed = True
            zha_info = replace(
                zha_info,
                local_protection_enabled=payload["localProtection"].lower()
                in {"enabled", "enabled (default)"},
            )

        if "doubleTapClearNotifications" in payload:
            zha_info_changed = True
            zha_info = replace(
                zha_info,
                double_tap_clear_notifications_disabled=payload[
                    "doubleTapClearNotifications"
                ].lower()
                == "disabled",
            )

        if zha_info_changed:
            self.store_switch_info(
                switch_id,
                integration_info=zha_info,
            )

    async def _handle_generic_event(
        self,
        *,
        command: str,
        switch_id: SwitchId,
        integration: Integration,
    ) -> None:
        from_state = self.switch_info(switch_id)
        led_config_source = from_state.led_config_source
        led_config = [*from_state.led_config]
        all_clear = False
        is_valid_dismissal = False
        dismissal_blocked = False
        index = 0

        if not led_config or not led_config_source:
            _LOGGER.warning(
                "missing LED config and/or source for dismissal on switch %s; "
                "skipping processing %s event %s",
                switch_id,
                str(integration).upper(),
                command,
                stack_info=True,
            )
            return

        _LOGGER.debug(
            "processing %s event %s on %s", str(integration).upper(), command, switch_id
        )

        if match := re.search(r"_LED_(\d+)$", command):
            index = int(match[1]) - 1
        else:
            led_config = [CLEAR_CONFIG]

        try:
            led_config[index] = CLEAR_CONFIG
        except IndexError:
            _LOGGER.warning(
                "could not clear switch config at index %s on %s: %s",
                index,
                switch_id,
                led_config,
                stack_info=True,
                exc_info=True,
            )

        if _all_clear(led_config):
            led_config = [CLEAR_CONFIG]
            all_clear = True

        if all_clear and command in CONFIGURABLE_COMMANDS:
            local_protection = self._local_protection_enabled(switch_id)
            disable_clear_notification = self._double_tap_clear_notifications_disabled(
                switch_id
            )

            is_valid_dismissal = not disable_clear_notification
            led_changed_via_firmware = not local_protection
        else:
            is_valid_dismissal = True
            led_changed_via_firmware = True

        # stop processing this if it's not a valid dismissal, i.e. the
        # `switch.<name>_disable_config_2x_tap_to_clear_notifications` has been
        # enabled by the user.
        if not is_valid_dismissal:
            return

        # immediately store the change that's been made on the switch if the
        # switch actually updated itself.
        if led_changed_via_firmware:
            self.store_switch_info(
                switch_id,
                led_config_source=led_config_source,
                led_config=tuple(led_config),
            )

        if (
            all_clear
            and (active := self._first_on_for_switch(switch_id))
            and led_config_source.is_for_notification(active.slug)
        ):
            try:
                await self.dismiss_notification(
                    active.slug,
                    dismissal_command=command,
                    via_switch=switch_id,
                )
            except _DismissalBlocked:
                # if the switch's firmware changed the LEDs being displayed,
                # then it's currently showing as all clear. the `switch_info`
                # was updated earlier in this method to match the firmware's
                # change before invoking `dismiss_notification`. the update to
                # switch_info` ensures the stored info matches the switch for
                # all update/dismissal cases (this allows calculating what
                # cluster messages need to be sent to get the the correct
                # state). these changes need to be reversed if the dismissal was
                # blocked, and the switch needs to be transitioned to actually
                # show the prior state since it's currently showing as all
                # clear. note: it's possible that the LED config is missing in
                # the from state, i.e. the system restarted & the recorded state
                # for the switch doesn't match what's really on the switch.
                if led_changed_via_firmware:
                    self.store_switch_info(
                        switch_id,
                        led_config=from_state.led_config,
                        led_config_source=from_state.led_config_source,
                    )
                    await self._transition_switch(
                        switch_id,
                        from_config=(),
                        to_config=from_state.led_config,
                    )
                dismissal_blocked = True
        elif all_clear:
            await self._switch_apply_notification_or_override(
                switch_id,
                exclude={LEDConfigSourceType.OVERRIDE},
                dismissal_command=command,
                log_context=f"dismissed via {switch_id}",
            )

        # clear the expiration after successfully dismissing
        if not dismissal_blocked:
            self.store_switch_info(
                switch_id,
                expiration=self._cancel_expiration(
                    self.switch_info(switch_id).expiration,
                    log_context=switch_id,
                ),
            )

    @staticmethod
    def _expire_leds(
        led_config: tuple[LEDConfig, ...],
        expiration: ExpirationInfo,
        now: dt.datetime,
    ) -> tuple[LEDConfig, ...]:
        result = []
        started_at = expiration.started_at

        assert started_at is not None, "expiration is missing start time"

        for idx, led in enumerate(led_config):
            led_expires_at = None
            led_is_expired = False

            if led.duration is not None:
                led_expires_at = started_at + dt.timedelta(seconds=led.duration)
                led_is_expired = led_expires_at <= now

            _LOGGER.log(
                TRACE,
                "LED %s: led_is_expired=%s, duration=%s, expires_at=%s, started=%s, now=%s",
                idx,
                led_is_expired,
                led.duration,
                led_expires_at,
                started_at,
                now,
            )

            if led_is_expired:
                result.append(replace(CLEAR_CONFIG, duration=ALREADY_EXPIRED))
            else:
                result.append(led)

        return tuple(result)

    @callback
    async def _async_handle_notification_expired(
        self,
        slug: Slug,
        led_config: tuple[LEDConfig, ...],
        now: dt.datetime,
    ) -> None:
        _LOGGER.debug("notification expired via timer (%s)", slug)

        coordinator = self._coordinators[slug]
        info = self.notification_info(slug)
        switches = [
            switch_id
            for switch_id in coordinator.config_entry.data[CONF_SWITCH_ENTITIES]
            if (switch_info := self.switch_info(switch_id))
            and (led_config_source := switch_info.led_config_source)
            and (led_config_source.is_for_notification(slug))
        ]

        await self._async_handle_expiration(
            now=now,
            expiration=info.expiration,
            switches=switches,
            led_config=led_config,
            store_expiration=partial(self.store_notification_info, slug),
            dismiss=partial(self.dismiss_notification, slug),
            handle_expired=partial(self._async_handle_notification_expired, slug),
            log_context=slug,
        )

    @callback
    async def _async_handle_switch_override_expired(
        self,
        switch_id: SwitchId,
        led_config: tuple[LEDConfig, ...],
        now: dt.datetime,
    ) -> None:
        _LOGGER.debug("switch override expired via timer (%s)", switch_id)

        await self._async_handle_expiration(
            now=now,
            expiration=self.switch_info(switch_id).expiration,
            switches=[switch_id],
            led_config=led_config,
            store_expiration=partial(self.store_switch_info, switch_id),
            dismiss=partial(
                self._switch_apply_notification_or_override,
                switch_id,
                exclude={LEDConfigSourceType.OVERRIDE},
                log_context=f"override expired for {switch_id}",
            ),
            handle_expired=partial(
                self._async_handle_switch_override_expired, switch_id
            ),
            log_context=switch_id,
        )

    async def _async_handle_expiration(
        self,
        *,
        now: dt.datetime,
        expiration: ExpirationInfo,
        switches: list[SwitchId],
        led_config: tuple[LEDConfig, ...],
        store_expiration: _ExpirationHandlerStoreExpirationCall,
        dismiss: _ExpirationHandlerDismissCall,
        handle_expired: _ExpirationHandler,
        log_context: str,
    ) -> None:
        _LOGGER.log(TRACE, "LED config before update: %s", led_config)
        led_config = self._expire_leds(led_config, expiration, now)
        _LOGGER.log(TRACE, "LED config after update: %s", led_config)

        if _all_clear(led_config):
            store_expiration(
                expiration=self._cancel_expiration(
                    expiration,
                    log_context=log_context,
                ),
            )
            await dismiss(dismissal_command="lampie_timer")
        else:
            store_expiration(
                expiration=self._schedule_fallback_expiration(
                    expiration,
                    led_config,
                    handle_expired,
                    is_starting=False,
                    has_expiration_messaging=self._any_has_expiration_messaging(
                        switches
                    ),
                    log_context=f"partial-expiry {log_context}",
                ),
            )

            # loop while waiting sequentially to avoid flooding the network
            for switch_id in switches:
                switch_info = self.switch_info(switch_id)
                led_config_source = (
                    switch_info.led_config_source if switch_info else None
                )

                await self._switch_apply_notification_or_override(
                    switch_id,
                    led_config_source=led_config_source,
                    led_config=led_config,
                    strip_durations_for_firmware=True,
                    log_context=f"partial-expiry {log_context}",
                )

    def _schedule_fallback_expiration(
        self,
        expiration: ExpirationInfo,
        led_config: tuple[LEDConfig, ...],
        callback: _ExpirationHandler,
        *,
        is_starting: bool = True,
        has_expiration_messaging: bool,
        log_context: str,
    ) -> ExpirationInfo:
        started_at = expiration.started_at
        expires_at = None
        cancel_listener = None

        if is_starting or started_at is None:
            started_at = dt_util.now()

            _LOGGER.debug("start time recorded (%s=%s)", log_context, started_at)

        durations: list[int] = [
            item.duration
            for item in led_config
            if item.duration is not None
            and item.duration != ALREADY_EXPIRED
            and (
                not has_expiration_messaging
                or self._firmware_duration(item.duration) is None
            )
        ]
        duration: int | None = min(durations) if durations else None

        if duration:
            expires_at = started_at + dt.timedelta(seconds=duration)
            cancel_listener = evt.async_track_point_in_time(
                self._hass,
                partial(callback, led_config),
                expires_at,
            )
            _LOGGER.debug("next expiration scheduled (%s:%s)", log_context, expires_at)

        self._cancel_expiration(expiration, log_context=log_context)

        return replace(
            expiration,
            started_at=started_at,
            expires_at=expires_at,
            cancel_listener=cancel_listener,
        )

    @classmethod
    def _cancel_expiration(
        cls,
        expiration: ExpirationInfo,
        *,
        log_context: str,
    ) -> ExpirationInfo:
        needs_clearing = False

        if expiration.started_at:
            needs_clearing = True
            _LOGGER.debug("start time cleared (%s)", log_context)

        if cancel_listener := expiration.cancel_listener:
            needs_clearing = True
            cancel_listener()
            _LOGGER.debug("expiration timer canceled (%s)", log_context)

        if needs_clearing:
            expiration = ExpirationInfo()

        return expiration

    async def _update_references(self) -> None:
        processed_slugs: list[Slug] = []
        processed_switches: set[SwitchId] = set()

        for coordinator in self._coordinators.values():
            slug: Slug = coordinator.slug
            entry = coordinator.config_entry
            switch_ids: list[SwitchId] = entry.data[CONF_SWITCH_ENTITIES]
            switch_priorities: dict[SwitchId, list[Slug]] = entry.data.get(
                CONF_PRIORITY, {}
            )

            for switch_id in switch_ids:
                try:
                    await self._ensure_switch_setup_completed(switch_id)
                except _UnknownIntegrationError:
                    _LOGGER.exception(
                        "ignoring switch %s: could not to a valid integration",
                        switch_id,
                    )
                    continue

                switch_info = self.switch_info(switch_id)
                priorities = switch_priorities.get(switch_id) or [slug]
                expected = [*switch_info.priorities]

                if switch_id in processed_switches and expected != priorities:
                    _LOGGER.warning(
                        "priorities mismatch found for %s in %s: "
                        "%s (previously seen) != %s (for %s), loading order: %s",
                        switch_id,
                        slug,
                        expected,
                        priorities,
                        slug,
                        ", ".join([repr(slug) for slug in processed_slugs]),
                    )

                if switch_id in processed_switches:
                    continue

                self._switches[switch_id] = replace(
                    switch_info,
                    priorities=tuple(priorities),
                )
                processed_switches.add(switch_id)

            processed_slugs.append(slug)

    @classmethod
    def _switch_integration(cls, device: dr.DeviceEntry) -> Integration | None:
        id_tuple = next(iter(device.identifiers))
        domain = id_tuple[0]
        return SWITCH_INTEGRATIONS.get(domain)


def _all_clear(led_config: Sequence[LEDConfig]) -> bool:
    return all(item.effect == Effect.CLEAR for item in led_config)
