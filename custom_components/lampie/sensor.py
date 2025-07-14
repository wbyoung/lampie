"""Support for Lampie sensors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
from statistics import mean
from typing import Literal

from homeassistant.components.sensor import (
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.const import PERCENTAGE, STATE_UNAVAILABLE, STATE_UNKNOWN, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import (
    ExtraStoredData,
    RestoredExtraData,
    RestoreEntity,
)

from .const import (
    ATTR_BRIGHTNESS,
    ATTR_COLOR,
    ATTR_DURATION,
    ATTR_EFFECT,
    ATTR_EXPIRES_AT,
    ATTR_INDIVIDUAL,
    ATTR_NOTIFICATION,
    ATTR_STARTED_AT,
    CONF_SWITCH_ENTITIES,
)
from .entity import LampieDistributedEntity, LampieEntityDescription
from .helpers import is_primary_for_switch, lookup_device_info
from .types import LampieConfigEntry, LampieSwitchInfo, LEDConfig, LEDConfigSource

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class LampieSensorDescription(
    SensorEntityDescription, LampieEntityDescription[LampieSwitchInfo]
):
    """Class describing Lampie sensor entities."""

    is_switch_info_restore_source: bool = False


def _combined[T](
    attr: str | Callable[[LEDConfig], T], fn: Callable[[list[T]], T]
) -> Callable[[LampieSwitchInfo], T | None]:
    getter = attr if callable(attr) else lambda item: item.to_dict()[attr]

    def value_fn(data: LampieSwitchInfo) -> T | None:
        if not data.led_config:
            return None
        return fn([getter(item) for item in data.led_config])

    return value_fn


def _unique[T](items: list[T]) -> T | Literal["multi"]:
    unique = set(items)

    if len(unique) == 1:
        return next(iter(unique))
    return "multi"


def _multi[T](
    data: LampieSwitchInfo, attr: str | Callable[[LEDConfig], T]
) -> dict[str, list[T]]:
    getter = attr if callable(attr) else lambda item: item.to_dict()[attr]

    if not data.led_config or len(data.led_config) == 1:
        return {}
    return {ATTR_INDIVIDUAL: [getter(item) for item in data.led_config]}


SENSOR_TYPES: tuple[SensorEntityDescription, ...] = (
    LampieSensorDescription(
        key=ATTR_NOTIFICATION,
        translation_key=ATTR_NOTIFICATION,
        is_switch_info_restore_source=True,
        value_fn=lambda data: data.led_config_source.value
        if data.led_config_source
        else None,
        icon="mdi:circle-box",
    ),
    LampieSensorDescription(
        key=ATTR_BRIGHTNESS,
        translation_key=ATTR_BRIGHTNESS,
        value_fn=_combined(lambda item: item.brightness, mean),
        attrs_fn=lambda data: _multi(data, lambda item: item.brightness),
        suggested_display_precision=0,
        native_unit_of_measurement=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:timer",
    ),
    LampieSensorDescription(
        key=ATTR_COLOR,
        translation_key=ATTR_COLOR,
        value_fn=_combined("color", _unique),
        attrs_fn=lambda data: _multi(data, "color"),
        icon="mdi:palette",
    ),
    LampieSensorDescription(
        key=ATTR_DURATION,
        translation_key=ATTR_DURATION,
        value_fn=_combined(
            "duration", lambda items: None if None in items else max(items)
        ),
        attrs_fn=lambda data: {
            **_multi(data, "duration"),
            **(
                (
                    (expiration := data.expiration)
                    and (started_at := expiration.started_at)
                    and (expires_at := expiration.expires_at)
                    and {
                        ATTR_STARTED_AT: started_at.isoformat(),
                        ATTR_EXPIRES_AT: expires_at.isoformat(),
                    }
                )
                or {}
            ),
        },
        native_unit_of_measurement=UnitOfTime.SECONDS,
        state_class=SensorStateClass.MEASUREMENT,
        icon="mdi:timer",
    ),
    LampieSensorDescription(
        key=ATTR_EFFECT,
        translation_key=ATTR_EFFECT,
        value_fn=_combined("effect", _unique),
        attrs_fn=lambda data: _multi(data, "effect"),
        icon="mdi:lightning-bolt",
    ),
)


async def async_setup_entry(  # noqa: RUF029
    hass: HomeAssistant,
    entry: LampieConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Lampie sensor entities based on a config entry."""
    coordinator = entry.runtime_data.coordinator

    entities: list[LampieSensor] = [
        LampieSensor(
            description=description,
            coordinator=coordinator,
            switch_id=switch_id,
            switch_device_info=device_info,
        )
        for description in SENSOR_TYPES
        for switch_id in entry.data[CONF_SWITCH_ENTITIES]
        if is_primary_for_switch(entry, switch_id)
        and (device_info := lookup_device_info(hass, entry, switch_id))
    ]

    async_add_entities(entities)


class LampieSensor(
    LampieDistributedEntity[LampieSwitchInfo],
    SensorEntity,
    RestoreEntity,
):
    """Lampie sensor."""

    async def async_added_to_hass(self) -> None:
        """Restore state."""
        await super().async_added_to_hass()

        if (
            self.entity_description.is_switch_info_restore_source
            and (last_state := await self.async_get_last_state()) is not None
            and last_state.state not in {STATE_UNKNOWN, STATE_UNAVAILABLE}
            and (extra_data := await self.async_get_last_extra_data()) is not None
        ):
            data = extra_data.as_dict()
            led_config = data.get("config", [])
            led_config_source = data.get("source")
            self.orchestrator.store_switch_info(
                self.switch_id,
                led_config=tuple(
                    LEDConfig.from_dict(item) for item in led_config or []
                ),
                led_config_source=LEDConfigSource.from_dict(led_config_source)
                if led_config_source is not None
                else LEDConfigSource(None),
            )

    @property
    def extra_restore_state_data(self) -> ExtraStoredData | None:
        """Save state."""

        if not self.entity_description.is_switch_info_restore_source:
            return None

        led_config = self.data.led_config
        led_config_source = self.data.led_config_source
        data = {
            "config": [item.to_dict() for item in led_config]
            if led_config is not None
            else None,
            "source": led_config_source.to_dict()
            if led_config_source is not None
            else None,
        }
        return RestoredExtraData(data) if data is not None else None

    @property
    def native_value(self) -> str | float | None:
        """Return native sensor value."""
        return self.entity_description.value_fn(self.data)

    @property
    def data(self) -> LampieSwitchInfo:
        """Get switch specific data from coordinator."""
        return self.orchestrator.switch_info(self.switch_id)
