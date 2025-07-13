"""Config flow for Lampie integration."""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager, suppress
import logging
from typing import Any, Final, NamedTuple

from homeassistant.components.fan import DOMAIN as FAN_DOMAIN
from homeassistant.components.light import DOMAIN as LIGHT_DOMAIN
from homeassistant.components.script import DOMAIN as SCRIPT_DOMAIN
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigEntryState,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import section
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.selector import (
    EntitySelector,
    EntitySelectorConfig,
    ObjectSelector,
    ObjectSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)
from homeassistant.util import slugify
import voluptuous as vol

from .const import (
    CONF_COLOR,
    CONF_DISMISS_ACTION,
    CONF_DURATION,
    CONF_EFFECT,
    CONF_END_ACTION,
    CONF_LED_CONFIG,
    CONF_NAME,
    CONF_PRIORITY,
    CONF_START_ACTION,
    CONF_SWITCH_ENTITIES,
    DOMAIN,
    INOVELLI_MODELS,
)
from .types import Color, Effect, InvalidColor, LampieConfigEntryRuntimeData, LEDConfig

_LOGGER = logging.getLogger(__name__)

SECTION_ADVANCED_OPTIONS: Final = "advanced_options"
SECTION_ADVANCED_ATTRS: Final = {
    CONF_LED_CONFIG,
    CONF_START_ACTION,
    CONF_END_ACTION,
    CONF_DISMISS_ACTION,
}


class Overlap(NamedTuple):
    """Overlap of Lampie config entries on a given switch."""

    switch_id: str
    slugs: list[str]


class LampieConfigFlow(ConfigFlow, domain=DOMAIN):  # type: ignore[call-arg]
    """Handle a Lampie config flow."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize."""
        self.flow_coordinator = LampieFlowCoordinator(self)
        super().__init__()

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Get the options flow for this handler.

        Returns:
            The options flow.
        """
        return LampieOptionsFlow(config_entry)

    async def async_step_user(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> ConfigFlowResult:
        """Handle a flow initialized by the user.

        Returns:
            The config flow result.
        """
        return await self.flow_coordinator.async_step_init(user_input, step_id="user")

    async def async_step_priority(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> ConfigFlowResult:
        """Handle the priority selection step(s).

        Returns:
            The config flow result.
        """
        return await self.flow_coordinator.async_step_priority(user_input)


class LampieOptionsFlow(OptionsFlow):
    """Handle a option flow."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize."""
        self.flow_coordinator = LampieFlowCoordinator(self, config_entry=config_entry)
        super().__init__()

    async def async_step_init(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> ConfigFlowResult:
        """Handle options flow.

        Returns:
            The config flow result.
        """
        return await self.flow_coordinator.async_step_init(user_input)

    async def async_step_priority(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> ConfigFlowResult:
        """Handle the priority selection step(s).

        Returns:
            The config flow result.
        """
        return await self.flow_coordinator.async_step_priority(user_input)

    def update_entry(self, entry_data: dict[str, Any]) -> None:
        self.hass.config_entries.async_update_entry(
            self.config_entry,
            data={**entry_data},
        )


class LampieFlowCoordinator:
    """Coordinator to share logic of config entry & options flows."""

    def __init__(
        self,
        flow: LampieConfigFlow | LampieOptionsFlow,
        *,
        config_entry: ConfigEntry | None = None,
    ) -> None:
        """Initialize."""
        self.flow_source = flow
        self.config_entry = config_entry
        self.is_options_flow = config_entry is not None
        self.switch_overlaps: deque[Overlap] = deque()
        self.title = self.config_entry.title if self.config_entry else None

        super().__init__()

    @property
    def hass(self) -> HomeAssistant:
        return self.flow_source.hass

    def _initial_data(self) -> dict[str, Any]:
        return dict(self.config_entry.data) if self.config_entry else {}

    def _other_entries(self) -> list[ConfigEntry]:
        return [
            entry
            for entry in self.hass.config_entries.async_entries(DOMAIN)
            if entry is not self.config_entry
        ]

    def _switch_overlaps(self) -> deque[Overlap]:
        result: deque[Overlap] = deque()
        selected_switches = self.entry_data[CONF_SWITCH_ENTITIES]
        self_slug = slugify(self.title)

        for switch_id in sorted(selected_switches):
            matches = [
                slugify(entry.title)
                for entry in self._other_entries()
                if switch_id in entry.data[CONF_SWITCH_ENTITIES]
            ]
            if matches:
                result.append(Overlap(switch_id, [self_slug, *matches]))

        return result

    def _schema_for_init(self, user_input: dict[str, Any] | None = None) -> vol.Schema:
        hass = self.hass
        device_registry = dr.async_get(hass)
        entity_registry = er.async_get(hass)

        if user_input is None:
            user_input = {**self._initial_data()}
            user_input[SECTION_ADVANCED_OPTIONS] = _pop(
                user_input,
                lambda key, _: key in SECTION_ADVANCED_ATTRS,
            )

        advanced_options = user_input.get(SECTION_ADVANCED_OPTIONS, {})

        return self.flow_source.add_suggested_values_to_schema(
            vol.Schema(
                (
                    {}
                    if self.is_options_flow
                    else {
                        vol.Required(
                            CONF_NAME,
                        ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
                    }
                )
                | {
                    vol.Optional(
                        CONF_COLOR,
                    ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
                    vol.Optional(
                        CONF_EFFECT,
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=[
                                SelectOptionDict(
                                    value=str(effect.name.lower()),
                                    label=" ".join(
                                        effect.name.lower().capitalize().split("_")
                                    ),
                                )
                                for effect in Effect
                            ],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_DURATION,
                    ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
                    vol.Required(
                        CONF_SWITCH_ENTITIES,
                    ): EntitySelector(
                        EntitySelectorConfig(
                            domain=[LIGHT_DOMAIN, FAN_DOMAIN],
                            include_entities=[
                                entity_id
                                for entity_id in entity_registry.entities
                                if _is_inovelli_switch(
                                    device_registry,
                                    entity_registry,
                                    entity_id,
                                )
                            ],
                            multiple=True,
                        ),
                    ),
                    vol.Required(SECTION_ADVANCED_OPTIONS): section(
                        vol.Schema(
                            {
                                vol.Optional(
                                    CONF_START_ACTION,
                                ): EntitySelector(
                                    EntitySelectorConfig(domain=[SCRIPT_DOMAIN]),
                                ),
                                vol.Optional(
                                    CONF_END_ACTION,
                                ): EntitySelector(
                                    EntitySelectorConfig(domain=[SCRIPT_DOMAIN]),
                                ),
                                vol.Optional(
                                    CONF_DISMISS_ACTION,
                                ): EntitySelector(
                                    EntitySelectorConfig(domain=[SCRIPT_DOMAIN]),
                                ),
                                vol.Optional(
                                    CONF_LED_CONFIG,
                                ): ObjectSelector(ObjectSelectorConfig()),
                            }
                        ),
                        {
                            "collapsed": all(
                                advanced_options.get(option) is None
                                for option in SECTION_ADVANCED_ATTRS
                            )
                        },
                    ),
                }
            ),
            user_input,
        )

    def _schema_for_priority(
        self, user_input: dict[str, Any] | None = None
    ) -> vol.Schema:
        if user_input is None:
            current_overlap = self.switch_overlaps[0]
            initial_priorities = self._initial_data().get(CONF_PRIORITY, {})
            user_input = {
                CONF_PRIORITY: initial_priorities.get(
                    current_overlap.switch_id, current_overlap.slugs
                )
            }

        return vol.Schema(
            {
                vol.Required(
                    CONF_PRIORITY,
                    default=user_input.get(CONF_PRIORITY),
                ): ObjectSelector(ObjectSelectorConfig())
            }
        )

    async def async_step_init(
        self,
        user_input: dict[str, Any] | None = None,
        *,
        step_id: str = "init",
    ) -> ConfigFlowResult:
        """Handle the start of a flow.

        Returns:
            The config flow result.
        """
        errors: dict[str, str] = {}
        description_placeholders: dict[str, str] = {}

        if self.title:
            description_placeholders["config_title"] = self.title

        if user_input is not None:
            color = user_input.get(CONF_COLOR, "")
            effect = user_input.get(CONF_EFFECT, "")
            advanced_options = user_input.get(SECTION_ADVANCED_OPTIONS, {})
            led_config = advanced_options.get(CONF_LED_CONFIG, [])
            duration = user_input.get(CONF_DURATION)

            if not color and not led_config:
                errors[CONF_COLOR] = "missing_color"
            if not effect and not led_config:
                errors[CONF_EFFECT] = "missing_effect"

            if color:
                self._validate_color(user_input, errors, description_placeholders)

            if led_config:
                self._validate_led_config(user_input, errors, description_placeholders)

            if duration:
                self._validate_duration(user_input, errors, description_placeholders)

        if user_input is not None and not errors:
            advanced_options = user_input.pop(SECTION_ADVANCED_OPTIONS)

            self.entry_data = {**user_input, **advanced_options}

            if self.entry_data.get(CONF_LED_CONFIG):
                self.entry_data.pop(CONF_COLOR, None)
                self.entry_data.pop(CONF_EFFECT, None)
                self.entry_data.pop(CONF_DURATION, None)
            else:
                self.entry_data.pop(CONF_LED_CONFIG, None)

            if not self.entry_data.get(CONF_DURATION):
                self.entry_data.pop(CONF_DURATION, None)

            if not self.is_options_flow:
                self.title = self._extract_name_to_unique_title()

            self.switch_overlaps = self._switch_overlaps()
            if self.switch_overlaps:
                return await self.async_step_priority()
            return await self._create_entry()

        return self.flow_source.async_show_form(
            step_id=step_id,
            data_schema=self._schema_for_init(user_input),
            errors=errors,
            description_placeholders=description_placeholders,
            last_step=False,
        )

    @classmethod
    def _validate_color(
        cls,
        user_input: dict[str, Any],
        errors: dict[str, str],
        description_placeholders: dict[str, str],
    ) -> None:
        color = user_input[CONF_COLOR]
        with suppress(ValueError):
            color = user_input[CONF_COLOR] = int(color)
        try:
            Color.color_number(color)
        except InvalidColor as e:
            errors[CONF_COLOR] = f"invalid_color_{e.reason}"
            description_placeholders["color"] = str(color)

    @classmethod
    def _validate_led_config(
        cls,
        user_input: dict[str, Any],
        errors: dict[str, str],
        description_placeholders: dict[str, str],
    ) -> None:
        advanced_options = user_input.get(SECTION_ADVANCED_OPTIONS, {})
        led_config = advanced_options.get(CONF_LED_CONFIG, [])

        for key in [CONF_COLOR, CONF_EFFECT, CONF_DURATION]:
            if user_input.get(key):
                errors[SECTION_ADVANCED_OPTIONS] = "invalid_led_config_override"
                description_placeholders["key"] = key
                break

        if not errors.get(SECTION_ADVANCED_OPTIONS):
            if not _is_listlike(led_config):
                errors[SECTION_ADVANCED_OPTIONS] = "invalid_led_config_type"
                description_placeholders["type"] = type(led_config).__name__
            elif len(led_config) != 7:
                errors[SECTION_ADVANCED_OPTIONS] = "invalid_led_config_length"
            else:
                for idx, item in enumerate(led_config):
                    try:
                        LEDConfig.from_config(item)
                    except InvalidColor as e:
                        if e.reason == "type":
                            errors[SECTION_ADVANCED_OPTIONS] = "invalid_color_type"
                            description_placeholders["type"] = str(e.context)
                            description_placeholders["index"] = str(idx)
                        else:
                            errors[SECTION_ADVANCED_OPTIONS] = (
                                "invalid_led_config_member"
                            )
                            description_placeholders["color"] = str(item)
                            description_placeholders["index"] = str(idx)
                        break

    @classmethod
    def _validate_duration(
        cls,
        user_input: dict[str, Any],
        errors: dict[str, str],
        description_placeholders: dict[str, str],
    ) -> None:
        duration = user_input[CONF_DURATION]
        try:
            LEDConfig.from_config({CONF_DURATION: duration})
        except vol.Invalid:
            errors[CONF_DURATION] = "invalid_duration"
            description_placeholders["duration"] = duration

    async def async_step_priority(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> ConfigFlowResult:
        """Handle the priority selection step(s).

        Returns:
            The config flow result.
        """
        switch_id = self.switch_overlaps[0].switch_id
        overlap = self.switch_overlaps[0]
        errors: dict[str, str] = {}
        description_placeholders = {
            "config_title": self.title,
            "switch_id": switch_id,
            "switch_name": self.hass.states.get(switch_id).name,
            "overlap": ", ".join(overlap.slugs),
        }

        if user_input is not None:
            expected_slugs = set(overlap.slugs)
            input_slugs = set(user_input[CONF_PRIORITY])
            missing = expected_slugs - input_slugs
            extra = input_slugs - expected_slugs

            if missing:
                errors["base"] = "missing_priorities"
                description_placeholders["missing_slugs"] = ", ".join(missing)

            if extra:
                errors["base"] = "extra_priorities"
                description_placeholders["extra_slugs"] = ", ".join(extra)

        if user_input is not None and not errors:
            priorities = self.entry_data.setdefault(CONF_PRIORITY, {})
            priorities.update({switch_id: user_input.pop(CONF_PRIORITY)})

            self.entry_data.update(user_input)
            self.switch_overlaps.popleft()

            if self.switch_overlaps:
                return await self.async_step_priority()
            return await self._create_entry()

        return self.flow_source.async_show_form(
            step_id="priority",
            data_schema=self._schema_for_priority(user_input),
            errors=errors,
            description_placeholders=description_placeholders,
            last_step=len(self.switch_overlaps) == 1,
        )

    def _extract_name_to_unique_title(self) -> str:
        slugified_existing_entry_titles = [
            slugify(entry.title) for entry in self._other_entries()
        ]

        title = self.entry_data.pop(CONF_NAME)
        possible_title: str = title
        tries = 1
        while slugify(possible_title) in slugified_existing_entry_titles:
            tries += 1
            possible_title = f"{title} {tries}"

        return possible_title

    def _clean_priorities(
        self, priorities: dict[str, Any] | None, *, entry: ConfigEntry | None = None
    ) -> None:
        """Remove unnecessary slugs from priorities lists.

        Remove keys from priorities when the list is a single slug that matches
        the config entry's slug; i.e. a `doors_open` and `windows_open`
        notifications have been added to `light.kitchen`. If removed from either
        config entry, both should end up with no priorities set.
        """
        entry_slug = slugify(entry.title if entry else self.title)

        if priorities:
            remove_keys = [
                key
                for key, value in priorities.items()
                if not value or value == [entry_slug]
            ]

            for key in remove_keys:
                del priorities[key]

    def _async_abort_entries_match(self) -> None:
        match_dict = {**self.entry_data}
        match_dict.pop(CONF_PRIORITY, None)

        self.flow_source._async_abort_entries_match(match_dict)  # noqa: SLF001

    @classmethod
    @asynccontextmanager
    async def _auto_reload_disabled(
        cls, other_entries: list[ConfigEntry]
    ) -> AsyncGenerator[None]:
        targets: list[LampieConfigEntryRuntimeData] = [
            entry.runtime_data
            for entry in other_entries
            if hasattr(entry, "runtime_data")
        ]

        for runtime_data in targets:
            runtime_data.auto_reload_enabled = False

        try:
            yield
        finally:
            for runtime_data in targets:
                runtime_data.auto_reload_enabled = True

    @asynccontextmanager
    async def _unloaded(self, other_entries: list[ConfigEntry]) -> AsyncGenerator[None]:
        other_entries = [  # filter to only the loaded entries
            entry for entry in other_entries if entry.state is ConfigEntryState.LOADED
        ]
        for entry in other_entries:
            await self.hass.config_entries.async_unload(entry.entry_id)

        yield

        for entry in other_entries:
            assert entry.state is ConfigEntryState.NOT_LOADED
            await self.hass.config_entries.async_setup(entry.entry_id)

    async def _update_other_entries_priorities(self) -> None:
        self_slug = slugify(self.title)
        other_entries = self._other_entries()
        initial_priorities = self._initial_data().get(CONF_PRIORITY, {})
        new_priorities = self.entry_data.get(CONF_PRIORITY, {})

        # get the keys we need to update, just those that have changed
        priority_keys = [
            key
            for key in (*initial_priorities.keys(), *new_priorities.keys())
            if initial_priorities.get(key, []) != new_priorities.get(key, [])
        ]

        other_entries = [
            entry
            for entry in other_entries
            if (set(priority_keys) & set(entry.data[CONF_SWITCH_ENTITIES]))
        ]

        async with (
            self._auto_reload_disabled(other_entries),
            self._unloaded(other_entries),
        ):
            for entry in other_entries:
                entry_priorities = {**entry.data.get(CONF_PRIORITY, {})}
                for priority_key in priority_keys:
                    if priority_key not in entry.data[CONF_SWITCH_ENTITIES]:
                        continue

                    new_priority = new_priorities.get(priority_key)

                    # when the priority is being removed from the current config
                    # entry, remove it from the list in the other config entry
                    # rather than just updating the list.
                    if not new_priority:
                        new_priority = entry_priorities.pop(priority_key, [])
                        new_priority = [
                            slug for slug in new_priority if slug != self_slug
                        ]

                    entry_priorities[priority_key] = new_priority

                    self._clean_priorities(entry_priorities, entry=entry)

                self.hass.config_entries.async_update_entry(
                    entry,
                    data={**entry.data, CONF_PRIORITY: entry_priorities},
                )

    async def _create_entry(self) -> ConfigFlowResult:
        entry_data = self.entry_data

        if not self.is_options_flow:
            self._async_abort_entries_match()

        await self._update_other_entries_priorities()

        if self.is_options_flow:
            self._clean_priorities(entry_data.get(CONF_PRIORITY))
            self.flow_source.update_entry(entry_data)
            entry_data = {}

        return self.flow_source.async_create_entry(title=self.title, data=entry_data)


def _is_inovelli_switch(
    device_registry: dr.DeviceRegistry,
    entity_registry: er.EntityRegistry,
    entity_id: str,
) -> bool:
    """Check if entity_id should be included for dependent entities.

    Determine if an entity_id represents an entity with a `state_class` of
    `total_increasing` and a `unit_of_measurement` of `km`.

    Returns:
        A flag indicating if the entity should be included.
    """
    return bool(
        (entity := entity_registry.async_get(entity_id))
        and (device := device_registry.async_get(entity.device_id))
        and (model := device.model)
        and (model in INOVELLI_MODELS)
    )


def _is_listlike[T](value: T) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _pop[KeyT, ValueT](
    collection: dict[KeyT, ValueT], matcher: Callable[[KeyT, ValueT], bool]
) -> dict[KeyT, ValueT]:
    result = {key: value for key, value in collection.items() if matcher(key, value)}

    for key in result:
        collection.pop(key)

    return result
