"""Lampie helpers."""

from __future__ import annotations

import logging

from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.device import async_device_info_to_link_from_entity
from homeassistant.util import slugify

from .const import CONF_PRIORITY
from .types import LampieConfigEntry

_LOGGER = logging.getLogger(__name__)


def is_primary_for_switch(entry: LampieConfigEntry, switch_id: str) -> bool:
    priorities = entry.data.get(CONF_PRIORITY, {}).get(switch_id)

    if not priorities:
        return True
    return bool(priorities[0] == slugify(entry.title))


def lookup_device_info(
    hass: HomeAssistant, entry: LampieConfigEntry, switch_id: str
) -> dr.DeviceInfo:
    device_info = async_device_info_to_link_from_entity(hass, switch_id)

    if not device_info:
        _LOGGER.warning(
            "skipping creation of sensors for %s on %s because an associated "
            "device could not be found",
            switch_id,
            entry.title,
        )

    return device_info
