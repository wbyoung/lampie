"""Support for the Lapmpie services."""

from functools import partial
from typing import Any, Final

from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
    callback,
)
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import config_validation as cv
import voluptuous as vol

from .const import ATTR_LED_CONFIG, ATTR_NAME, ATTR_NOTIFICATION, DOMAIN
from .orchestrator import LampieOrchestrator
from .types import (
    Color,
    Effect,
    InvalidColor,
    LEDConfig,
    LEDConfigSource,
    LEDConfigSourceType,
)

SERVICE_NAME_ACTIVATE: Final = "activate"
SERVICE_NAME_OVERRIDE: Final = "override"


def _led_config(
    value: Any,  # noqa: ANN401
) -> LEDConfig:
    try:
        return LEDConfig.from_config(value)
    except InvalidColor as e:
        msg = f"{value} is not a valid color; code: `{e.reason}`"
        raise vol.Invalid(msg) from e


def _led_configs(value: str | list | None) -> tuple[LEDConfig, ...] | None:
    if value is None:
        return value
    if isinstance(value, str):
        value = [value]
    return tuple(_led_config(item) for item in value)


SERVICE_SCHEMA_ACTIVATE: Final = vol.Schema(
    {
        vol.Required(ATTR_NOTIFICATION): vol.All(cv.ensure_list, [cv.string]),
        vol.Optional(ATTR_LED_CONFIG): _led_configs,
    },
)

SERVICE_SCHEMA_OVERRIDE: Final = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_ids,
        vol.Required(ATTR_LED_CONFIG): _led_configs,
        vol.Optional(ATTR_NAME): cv.string,
    },
)


async def _activate(
    call: ServiceCall,
    *,
    hass: HomeAssistant,
) -> ServiceResponse:
    slugs: list[str] = call.data[ATTR_NOTIFICATION]
    led_config: tuple[LEDConfig, ...] | None = call.data.get(ATTR_LED_CONFIG)
    led_config_source = (
        LEDConfigSource("lampie.activate", LEDConfigSourceType.SERVICE)
        if led_config is not None
        else None
    )

    orchestrator: LampieOrchestrator = hass.data[DOMAIN]

    for slug in slugs:
        if not orchestrator.has_notification(slug):
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="invalid_notification",
                translation_placeholders={
                    "slug": slug,
                },
            )

    for slug in slugs:
        await orchestrator.activate_notification(  # waiting in loop to avoid flooding the network
            slug,
            led_config=led_config,
            led_config_source=led_config_source,
        )


async def _override(
    call: ServiceCall,
    *,
    hass: HomeAssistant,
) -> ServiceResponse:
    name: str = call.data.get(ATTR_NAME) or f"{DOMAIN}.{SERVICE_NAME_OVERRIDE}"
    switch_ids: list[str] = call.data[ATTR_ENTITY_ID]
    led_config: tuple[LEDConfig, ...] | None = call.data[ATTR_LED_CONFIG]
    led_config_source = LEDConfigSource(name, LEDConfigSourceType.SERVICE)

    if led_config is None:  # none indicates a reset
        led_config = ()
        led_config_source = LEDConfigSource(None)
    elif not led_config:  # convert empty led config to clear config
        led_config = (LEDConfig(Color.BLUE, Effect.CLEAR),)

    orchestrator: LampieOrchestrator = hass.data[DOMAIN]

    for switch_id in switch_ids:
        await orchestrator.override_switch(  # waiting in loop to avoid flooding the network
            switch_id,
            led_config_source,
            led_config,
        )


@callback
def async_setup_services(hass: HomeAssistant) -> None:
    """Set up Lampie services."""
    hass.services.async_register(
        DOMAIN,
        SERVICE_NAME_ACTIVATE,
        partial(_activate, hass=hass),
        schema=SERVICE_SCHEMA_ACTIVATE,
        supports_response=SupportsResponse.NONE,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_NAME_OVERRIDE,
        partial(_override, hass=hass),
        schema=SERVICE_SCHEMA_OVERRIDE,
        supports_response=SupportsResponse.NONE,
    )
