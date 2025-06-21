"""Constants for the Lampie integration."""

from typing import Final

from homeassistant.const import ATTR_ENTITY_ID as _ATTR_ENTITY_ID

DOMAIN: Final = "lampie"

ATTR_BRIGHTNESS: Final = "brightness"
ATTR_COLOR: Final = "color"
ATTR_DURATION: Final = "duration"
ATTR_EFFECT: Final = "effect"
ATTR_ENTITY_ID: Final = _ATTR_ENTITY_ID
ATTR_EXPIRES_AT: Final = "expires_at"
ATTR_LED_CONFIG: Final = "leds"
ATTR_INDIVIDUAL: Final = "individual"
ATTR_NAME: Final = "name"
ATTR_NOTIFICATION: Final = "notification"
ATTR_STARTED_AT: Final = "started_at"
ATTR_TYPE: Final = "type"
ATTR_VALUE: Final = "value"

CONF_BRIGTNESS: Final = "brightness"
CONF_COLOR: Final = "color"
CONF_DISMISS_ACTION: Final = "dismiss_action"
CONF_DURATION: Final = "duration"
CONF_EFFECT: Final = "effect"
CONF_END_ACTION: Final = "end_action"
CONF_LED_CONFIG: Final = "led_config"
CONF_NAME: Final = "name"
CONF_PRIORITY: Final = "priority"
CONF_START_ACTION: Final = "start_action"
CONF_SWITCH_ENTITIES: Final = "switches"

INOVELLI_MODELS = {
    "VZM30-SN",  # switch
    "VZM31-SN",  # two in one switch/dimmer
    "VZM35-SN",  # fan switch
    "VZM36",  # canopy module
}

TRACE: Final = 5
