"""Lampie dataclasses and typing."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
import datetime as dt
from enum import Enum, IntEnum, StrEnum, auto
from typing import TYPE_CHECKING, Any, NotRequired, Self, TypedDict

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import CALLBACK_TYPE
from homeassistant.helpers import config_validation as cv

from .const import (
    ATTR_BRIGHTNESS,
    ATTR_COLOR,
    ATTR_DURATION,
    ATTR_EFFECT,
    ATTR_TYPE,
    ATTR_VALUE,
    CONF_BRIGTNESS,
    CONF_COLOR,
    CONF_DURATION,
    CONF_EFFECT,
)

if TYPE_CHECKING:
    from .coordinator import LampieUpdateCoordinator
    from .orchestrator import LampieOrchestrator

type Slug = str
type DeviceId = str
type EntityId = str
type SwitchId = EntityId
type LampieConfigEntry = ConfigEntry[LampieUpdateCoordinator]


class Color(IntEnum):
    """Available colors."""

    RED = 0
    BLUE = 170
    CYAN = 130
    GREEN = 90
    PINK = 230
    ORANGE = 25
    YELLOW = 45
    PURPLE = 200
    WHITE = 255

    @staticmethod
    def parse(value: str) -> Color:
        """Parse a color from a string.

        Returns:
            The color.

        Raises:
            InvalidColor: If the color name is invalid.
        """
        color: Color | None = getattr(Color, value.upper(), None)

        if color is None:
            raise InvalidColor(reason="name")

        return color

    @staticmethod
    def parse_or_validate_in_range(value: str | int) -> Color | int:
        """Get a color or number in the valid color range.

        Validate an input value and convert to a number that's in the valid
        color range.

        Returns:
            A number representing a color.

        Raises:
            InvalidColor: If the color is invalid because it is not a valid
                name, is not in range, or is the wrong type.
        """
        with suppress(ValueError, TypeError):
            value = int(value)

        if isinstance(value, str):
            return Color.parse(value)
        if isinstance(value, int):
            if value not in range(256):
                raise InvalidColor(reason="out_of_range")
            return value
        raise InvalidColor(reason="type", context=type(value).__name__)


class Effect(Enum):
    """Available effects."""

    OFF = 0
    SOLID = 1
    FAST_BLINK = 2
    SLOW_BLINK = 3
    PULSE = 4
    CHASE = 5
    OPEN_CLOSE = 6
    SMALL_TO_BIG = 7
    AURORA = 8
    SLOW_FALLING = 9
    MEDIUM_FALLING = 10
    FAST_FALLING = 11
    SLOW_RISING = 12
    MEDIUM_RISING = 13
    FAST_RISING = 14
    MEDIUM_BLINK = 15
    SLOW_CHASE = 16
    FAST_CHASE = 17
    FAST_SIREN = 18
    SLOW_SIREN = 19
    CLEAR = 255


@dataclass(frozen=True)
class LEDConfig:
    """LED configuration."""

    color: Color | int
    effect: Effect
    brightness: float = 100.0
    duration: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            ATTR_COLOR: self.color.name.lower()
            if isinstance(self.color, Color)
            else self.color,
            ATTR_EFFECT: self.effect.name.lower(),
            ATTR_BRIGHTNESS: self.brightness,
            ATTR_DURATION: self.duration,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            color=getattr(Color, data[ATTR_COLOR].upper())
            if isinstance(data[ATTR_COLOR], str)
            else data[ATTR_COLOR],
            effect=getattr(Effect, data[ATTR_EFFECT].upper()),
            brightness=float(data[ATTR_BRIGHTNESS]),
            duration=int(data[ATTR_DURATION])
            if data[ATTR_DURATION] is not None
            else None,
        )

    @staticmethod
    def from_config(config: str | Mapping[str, Any]) -> LEDConfig:
        if not isinstance(config, Mapping):
            config = {CONF_COLOR: str(config)}

        color = config.get(CONF_COLOR, Color.BLUE.name)
        color = Color.parse_or_validate_in_range(color)
        brightness: float = config.get(CONF_BRIGTNESS, 100.0)
        effect: Effect = getattr(
            Effect, config.get(CONF_EFFECT, Effect.SOLID.name).upper()
        )
        duration = config.get(CONF_DURATION)

        if duration is not None:
            try:
                duration = int(duration)
            except ValueError:
                duration = cv.time_period_str(duration).total_seconds()

        return LEDConfig(
            color=color,
            brightness=brightness,
            effect=effect,
            duration=duration,
        )


class LEDConfigSourceType(StrEnum):
    """LED configuration source type."""

    NOTIFICATION = auto()
    SERVICE = auto()


@dataclass(frozen=True)
class LEDConfigSource:
    """LED configuration source."""

    value: str | None
    type: LEDConfigSourceType = LEDConfigSourceType.NOTIFICATION

    def __str__(self) -> str:
        return self.type.name.lower() + (f":{self.value}" if self.value else "")

    def is_for_notification(self, slug: str) -> bool:
        return self.type == LEDConfigSourceType.NOTIFICATION and self.value == slug

    def to_dict(self) -> dict[str, Any]:
        return {
            ATTR_VALUE: self.value,
            ATTR_TYPE: self.type.name.lower(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            value=data[ATTR_VALUE],
            type=getattr(LEDConfigSourceType, data[ATTR_TYPE].upper()),
        )


@dataclass(frozen=True)
class InvalidColor(Exception):
    """Error for invalid color."""

    reason: str
    context: str | None = None
    index: int | None = None


@dataclass(frozen=True)
class ExpirationInfo:
    """Storage of expiration info."""

    started_at: dt.datetime | None = None
    expires_at: dt.datetime | None = None
    cancel_listener: CALLBACK_TYPE | None = None


@dataclass(frozen=True)
class LampieNotificationInfo:
    """Lampie notification data class."""

    notification_on: bool = False
    led_config_override: tuple[LEDConfig, ...] | None = None
    expiration: ExpirationInfo = field(default_factory=ExpirationInfo)


class LampieNotificationOptionsDict(TypedDict):
    """Lampie notification data class dict representation for kwargs."""

    notification_on: NotRequired[bool]
    led_config_override: NotRequired[tuple[LEDConfig, ...] | None]
    expiration: NotRequired[ExpirationInfo]


@dataclass(frozen=True)
class LampieSwitchInfo:
    """Lampie switch data class."""

    led_config: tuple[LEDConfig, ...]
    led_config_source: LEDConfigSource
    local_protection_id: EntityId | None = None
    disable_clear_notification_id: EntityId | None = None
    priorities: tuple[Slug, ...] = field(default_factory=tuple)
    expiration: ExpirationInfo = field(default_factory=ExpirationInfo)


class LampieSwitchOptionsDict(TypedDict):
    """Lampie switch data class dict representation for kwargs."""

    led_config: NotRequired[tuple[LEDConfig, ...]]
    led_config_source: NotRequired[LEDConfigSource]
    local_protection_id: NotRequired[EntityId | None]
    disable_clear_notification_id: NotRequired[EntityId | None]
    priorities: NotRequired[tuple[Slug, ...]]
    expiration: NotRequired[ExpirationInfo]


@dataclass
class LampieConfigEntryRuntimeData:
    """Runtime data definition."""

    orchestrator: LampieOrchestrator
    coordinator: LampieUpdateCoordinator
    auto_reload_enabled: bool = True
