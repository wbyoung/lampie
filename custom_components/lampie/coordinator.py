"""The Lampie coordinator."""

from __future__ import annotations

import logging

from homeassistant.core import Event, HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.issue_registry import IssueSeverity, async_create_issue
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import slugify

from .const import (
    CONF_COLOR,
    CONF_DURATION,
    CONF_EFFECT,
    CONF_LED_CONFIG,
    CONF_SWITCH_ENTITIES,
    DOMAIN,
)
from .types import LampieConfigEntry, LEDConfig

_LOGGER = logging.getLogger(__name__)


class LampieUpdateCoordinator(DataUpdateCoordinator[None]):
    """Class to manage Lampie updates."""

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: LampieConfigEntry,
    ) -> None:
        """Initialize."""
        super().__init__(
            hass,
            _LOGGER,
            config_entry=config_entry,
            name=config_entry.title,
            update_interval=None,
        )

        self.data = None
        self.slug = slugify(config_entry.title)
        self.led_config = tuple(
            LEDConfig.from_config(item)
            for item in (
                config_entry.data.get(
                    CONF_LED_CONFIG,
                    [
                        {
                            CONF_COLOR: config_entry.data.get(CONF_COLOR),
                            CONF_EFFECT: config_entry.data.get(CONF_EFFECT),
                            CONF_DURATION: config_entry.data.get(CONF_DURATION),
                        }
                    ],
                )
            )
        )

    async def _async_update_data(self) -> None:
        return self.data

    async def async_handle_switch_id_change(
        self,
        event: Event[er.EventEntityRegistryUpdatedData],
    ) -> None:
        """Fetch and process switch entity change event."""
        data = event.data
        if data["action"] == "remove":
            self._create_removed_switch_id_issue(data["entity_id"])

        if data["action"] == "update" and "entity_id" in data["changes"]:
            old_switch_id = data["old_entity_id"]
            new_switch_id = data["entity_id"]
            switch_ids = self.config_entry.data[CONF_SWITCH_ENTITIES]

            self.hass.config_entries.async_update_entry(
                self.config_entry,
                data={
                    **self.config_entry.data,
                    CONF_SWITCH_ENTITIES: [
                        new_switch_id if switch_id == old_switch_id else switch_id
                        for switch_id in switch_ids
                    ],
                },
            )

    def _create_removed_switch_id_issue(self, entity_id: str) -> None:
        """Create a repair issue for a removed switch entity."""
        async_create_issue(
            self.hass,
            DOMAIN,
            f"switch_entity_removed_{entity_id}",
            is_fixable=True,
            is_persistent=True,
            severity=IssueSeverity.WARNING,
            translation_key="switch_entity_removed",
            translation_placeholders={
                "entity_id": entity_id,
                "name": self.name,
            },
        )
