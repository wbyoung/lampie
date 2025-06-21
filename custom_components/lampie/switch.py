"""Support for Lampie switches."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from homeassistant.components.switch import SwitchEntity, SwitchEntityDescription
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import (
    ExtraStoredData,
    RestoredExtraData,
    RestoreEntity,
)

from .const import ATTR_EXPIRES_AT, ATTR_NOTIFICATION, ATTR_STARTED_AT
from .entity import LampieEntity, LampieEntityDescription
from .types import LampieConfigEntry, LampieNotificationInfo, LEDConfig

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class LampieSwitchDescription(
    SwitchEntityDescription, LampieEntityDescription[LampieNotificationInfo]
):
    """Class describing Lampie switch entities."""

    is_notification_info_restore_source: bool = False


SENSOR_TYPES: tuple[SwitchEntityDescription, ...] = (
    LampieSwitchDescription(
        key=ATTR_NOTIFICATION,
        translation_key=ATTR_NOTIFICATION,
        is_notification_info_restore_source=True,
        value_fn=lambda data: data.notification_on,
        attrs_fn=lambda data: (
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
        icon="mdi:circle-box",
    ),
)


async def async_setup_entry(  # noqa: RUF029
    hass: HomeAssistant,  # noqa: ARG001
    entry: LampieConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Lampie switch entities based on a config entry."""
    coordinator = entry.runtime_data.coordinator

    entities: list[LampieSwitch] = [
        LampieSwitch(
            description=description,
            coordinator=coordinator,
        )
        for description in SENSOR_TYPES
    ]

    async_add_entities(entities)


class LampieSwitch(
    LampieEntity[LampieNotificationInfo],
    SwitchEntity,
    RestoreEntity,
):
    """Lampie switch."""

    async def async_added_to_hass(self) -> None:
        """Restore state."""
        await super().async_added_to_hass()

        if (
            self.entity_description.is_notification_info_restore_source
            and (last_state := await self.async_get_last_state()) is not None
            and last_state.state not in {STATE_UNKNOWN, STATE_UNAVAILABLE}
            and (extra_data := await self.async_get_last_extra_data()) is not None
        ):
            data = extra_data.as_dict()
            led_config_override = data.get("config_override")
            self.orchestrator.store_notification_info(
                self.coordinator.slug,
                notification_on=data.get("on", False),
                led_config_override=tuple(
                    LEDConfig.from_dict(item) for item in led_config_override
                )
                if led_config_override is not None
                else None,
            )

    @property
    def extra_restore_state_data(self) -> ExtraStoredData | None:
        """Save state."""

        if not self.entity_description.is_notification_info_restore_source:
            return None

        led_config_override = self.data.led_config_override
        data = {
            "on": self.data.notification_on,
            "config_override": [item.to_dict() for item in led_config_override]
            if led_config_override is not None
            else None,
        }
        return RestoredExtraData(data) if data is not None else None

    @property
    def data(self) -> LampieNotificationInfo:
        """Get data from coordinator."""
        return self.orchestrator.notification_info(self.coordinator.slug)

    @property
    def is_on(self) -> bool:
        """Return native sensor value."""
        return bool(self.entity_description.value_fn(self.data))

    async def async_turn_on(
        self,
        **kwargs,  # noqa: ARG002, ANN003
    ) -> None:
        await self.orchestrator.activate_notification(self.coordinator.slug)

    async def async_turn_off(
        self,
        **kwargs,  # noqa: ARG002, ANN003
    ) -> None:
        await self.orchestrator.dismiss_notification(self.coordinator.slug)
