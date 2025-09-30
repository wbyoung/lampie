"""The Lampie orchestrator."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import replace
import datetime as dt
from enum import IntEnum
from functools import partial
import logging
import re
from typing import TYPE_CHECKING, Any, Final, NamedTuple, Protocol, Unpack

from homeassistant.components.script import DOMAIN as SCRIPT_DOMAIN
from homeassistant.core import CALLBACK_TYPE, Event, HomeAssistant, callback
from homeassistant.helpers import entity_registry as er, event as evt
from homeassistant.helpers.device import (
    async_device_info_to_link_from_entity,
    async_entity_id_to_device_id,
)
from homeassistant.util import dt as dt_util

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
    LampieNotificationInfo,
    LampieNotificationOptionsDict,
    LampieSwitchInfo,
    LampieSwitchOptionsDict,
    LEDConfig,
    LEDConfigSource,
    LEDConfigSourceType,
    Slug,
    SwitchId,
)

if TYPE_CHECKING:
    from .coordinator import LampieUpdateCoordinator

type ZHAEventData = dict[str, Any]

_LOGGER = logging.getLogger(__name__)

ZHA_DOMAIN: Final = "zha"
ALREADY_EXPIRED: Final = 0

FIRMWARE_SECONDS_MAX = dt.timedelta(seconds=60).total_seconds()
FIRMWARE_MINUTES_MAX = dt.timedelta(minutes=60).total_seconds()
FIRMWARE_HOURS_MAX = dt.timedelta(hours=134).total_seconds()

CLEAR_CONFIG = LEDConfig(
    color=Color.BLUE,
    effect=Effect.CLEAR,
)

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


class _LEDMode(IntEnum):
    ALL = 1
    INDIVIDUAL = 3


class _StartScriptResult(NamedTuple):
    led_config: tuple[LEDConfig, ...] | None
    block_activation: bool


class _EndScriptResult(NamedTuple):
    block_dismissal: bool
    block_next: bool


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
        self._device_switches: dict[DeviceId, SwitchId] = {}
        self._cancel_zha_listener: CALLBACK_TYPE = hass.bus.async_listen(
            "zha_event",
            self._handle_zha_event,
            self._filter_zha_events,
        )

    def add_coordinator(self, coordinator: LampieUpdateCoordinator) -> None:
        self._coordinators[coordinator.slug] = coordinator
        self._update_references()

    def remove_coordinator(self, coordinator: LampieUpdateCoordinator) -> None:
        self._coordinators.pop(coordinator.slug)
        self._update_references()

    def teardown(self) -> bool:
        if len(self._coordinators) == 0:
            self._cancel_zha_listener()
            for key, expiration in (
                *((key, info.expiration) for key, info in self._notifications.items()),
                *((key, info.expiration) for key, info in self._switches.items()),
            ):
                self._cancel_expiration(expiration, log_context=f"teardown:{key}")
            return True
        return False

    def switch_info(self, switch_id: SwitchId) -> LampieSwitchInfo:
        if switch_id not in self._switches:
            entity_registry = er.async_get(self._hass)
            device_id = async_entity_id_to_device_id(self._hass, switch_id)
            entity_entries = er.async_entries_for_device(entity_registry, device_id)
            local_protection_id = None
            disable_clear_notification_id = None

            _LOGGER.debug(
                "searching %s related entries: %s",
                switch_id,
                [(entry.unique_id, entry.translation_key) for entry in entity_entries],
            )

            for entity_entry in entity_entries:
                if entity_entry.translation_key == "local_protection":
                    local_protection_id = entity_entry.entity_id
                if (
                    entity_entry.translation_key
                    == "disable_clear_notifications_double_tap"
                ):
                    disable_clear_notification_id = entity_entry.entity_id

            self._device_switches[device_id] = switch_id
            self._switches[switch_id] = LampieSwitchInfo(
                led_config=(),
                led_config_source=LEDConfigSource(None),
                local_protection_id=local_protection_id,
                disable_clear_notification_id=disable_clear_notification_id,
            )
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
        via_switch_firmware: bool = False,
    ) -> None:
        """Dismiss a notification.

        Perform all necessary actions and validations to dismiss a notification,
        and update all switches as necessary based on the outcome. Switches may
        be updated to display another notification or cleared.
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
            # set the switch back to what it was before if the dismissal was
            # blocked and the switch is already showing the clear config because
            # the dismissal was via the switch firmware. note: it's possible
            # that the LED config is missing in the from state, i.e. the system
            # restarted & the recorded state for the switch doesn't match what's
            # really on the switch.
            if (
                via_switch
                and via_switch_firmware
                and (prior_led_config := self.switch_info(via_switch).led_config)
            ):
                await self._transition_switch(
                    via_switch,
                    from_config=(),
                    to_config=prior_led_config,
                )
            return

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
                via_switch_firmware=via_switch_firmware and via_switch == switch_id,
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
        from_state = self.switch_info(switch_id)
        is_reset = led_config is None

        self.store_switch_info(
            switch_id,
            expiration=self._schedule_fallback_expiration(
                from_state.expiration,
                led_config or (),
                partial(self._async_handle_switch_override_expired, switch_id),
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
        via_switch_firmware: bool = False,
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
            via_switch_firmware: A flag for if this change was via the switch
                firmware which is passed along to the method that performs the
                switch transition to minimize updates.
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
            via_switch_firmware=via_switch_firmware,
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
        via_switch_firmware: bool = False,
    ) -> None:
        _LOGGER.log(
            TRACE, "transition_switch: %s; %s -> %s", switch_id, from_config, to_config
        )

        if via_switch_firmware:
            return

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

        from_params = [self._switch_command_led_params(led) for led in from_config]
        device_info = async_device_info_to_link_from_entity(self._hass, switch_id)
        id_tuple = next(iter(device_info["identifiers"]))
        updated_leds = []
        ieee = id_tuple[1]

        # actually apply the desired changes either for the full LED bar or for
        # individual lights within the bar. only apply changes when needed to
        # reduce the number of messages sent and how much the bars/lights flash.
        # no changes need to be applied if the changes were via the switch
        # firmware since the switch will already have cleared the notification.
        for idx, led in enumerate(led_config):
            params = self._switch_command_led_params(led)

            with suppress(IndexError):
                if params == from_params[idx]:
                    _LOGGER.log(TRACE, "skipping update of LED %s; no change", idx)
                    continue

            if led_mode == _LEDMode.INDIVIDUAL:
                params["led_number"] = idx
                updated_leds.append(str(idx))

            _LOGGER.log(TRACE, "update LED %s command: %r", idx, params)
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

        _LOGGER.debug(
            "issued commands to update %s leds: %s",
            switch_id,
            "all" if led_mode == _LEDMode.ALL else ", ".join(updated_leds),
        )

    @classmethod
    def _switch_command_led_params(cls, led: LEDConfig) -> dict[str, Any]:
        firmware_duration = cls._firmware_duration(led.duration)

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

    @classmethod
    def _firmware_duration(cls, seconds: int | None) -> int | None:
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
        return (
            event_data["device_id"] in self._device_switches
            and event_data["command"] in DISMISSAL_COMMANDS
        )

    @callback
    async def _handle_zha_event(self, event: Event[ZHAEventData]) -> None:
        hass = self._hass
        command = event.data["command"]
        device_id = event.data["device_id"]
        switch_id = self._device_switches[device_id]
        from_state = self.switch_info(switch_id)
        led_config_source = from_state.led_config_source
        led_config = [*from_state.led_config]
        all_clear = False
        is_valid_dismissal = False
        index = 0

        if not led_config or not led_config_source:
            _LOGGER.warning(
                "missing LED config and/or source for dismissal on switch %s; "
                "skipping processing ZHA event %s",
                switch_id,
                command,
                stack_info=True,
            )
            return

        _LOGGER.debug("processing ZHA event %s on %s", command, switch_id)

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
            switch_info = self.switch_info(switch_id)
            local_protection = switch_info.local_protection_id and hass.states.is_state(
                switch_info and switch_info.local_protection_id, "on"
            )
            disable_clear_notification = (
                switch_info.disable_clear_notification_id
                and hass.states.is_state(
                    switch_info.disable_clear_notification_id, "on"
                )
            )

            is_valid_dismissal = not disable_clear_notification
            via_switch_firmware = not local_protection
        else:
            is_valid_dismissal = True
            via_switch_firmware = True

        # stop processing this if it's not a valid dismissal, i.e. the
        # `switch.<name>_disable_config_2x_tap_to_clear_notifications` has been
        # enabled by the user.
        if not is_valid_dismissal:
            return

        self.store_switch_info(
            switch_id,
            expiration=self._cancel_expiration(
                self.switch_info(switch_id).expiration,
                log_context=switch_id,
            ),
        )

        if (
            all_clear
            and (active := self._first_on_for_switch(switch_id))
            and led_config_source.is_for_notification(active.slug)
        ):
            await self.dismiss_notification(
                active.slug,
                dismissal_command=command,
                via_switch=switch_id,
                via_switch_firmware=via_switch_firmware,
            )
        elif all_clear:
            await self._switch_apply_notification_or_override(
                switch_id,
                exclude={LEDConfigSourceType.OVERRIDE},
                dismissal_command=command,
                via_switch_firmware=via_switch_firmware,
                log_context=f"dismissed via {switch_id}",
            )
        else:
            self.store_switch_info(
                switch_id,
                led_config_source=led_config_source,
                led_config=tuple(led_config),
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
            and self._firmware_duration(item.duration) is None
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

    def _update_references(self) -> None:
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
                priorities = switch_priorities.get(switch_id) or [slug]
                expected = [*self.switch_info(switch_id).priorities]

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
                    self.switch_info(switch_id),
                    priorities=tuple(priorities),
                )
                processed_switches.add(switch_id)

            processed_slugs.append(slug)


def _all_clear(led_config: Sequence[LEDConfig]) -> bool:
    return all(item.effect == Effect.CLEAR for item in led_config)
