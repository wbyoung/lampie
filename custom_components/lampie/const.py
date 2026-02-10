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

CONF_BRIGHTNESS: Final = "brightness"
CONF_COLOR: Final = "color"
CONF_DURATION: Final = "duration"
CONF_EFFECT: Final = "effect"
CONF_END_ACTION: Final = "end_action"
CONF_LED_CONFIG: Final = "led_config"
CONF_NAME: Final = "name"
CONF_PRIORITY: Final = "priority"
CONF_START_ACTION: Final = "start_action"
CONF_SWITCH_ENTITIES: Final = "switches"

INOVELLI_MODELS = {
    "LZW30-SN",  # red on/off switch
    "LZW31-SN",  # red dimmer
    "LZW36",  # red fan/light combo
    "VTM31-SN",  # white dimmer (2-in-1 switch/dimmer)
    "VTM35-SN",  # white fan
    "VZM30-SN",  # blue switch
    "VZM31-SN",  # blue 2-in-1 switch/dimmer
    "VZM35-SN",  # blue fan switch
    "VZM36",  # blue canopy module
    "VZW31-SN",  # red 2-in-1 dimmer
}


ZWAVE_EFFECT_PARAMETERS = {
    "LZW31-SN": 16,  # red dimmer
    "LZW30-SN": 8,  # red on/off switch
    "LZW36_light": 24,  # red fan/light combo
    "LZW36_fan": 25,  # red fan/light combo
    "VZW31-SN": 99,  # red 2-in-1 dimmer
    "VZW31-SN_individual_1": 64,
    "VZW31-SN_individual_2": 69,
    "VZW31-SN_individual_3": 74,
    "VZW31-SN_individual_4": 79,
    "VZW31-SN_individual_5": 84,
    "VZW31-SN_individual_6": 89,
    "VZW31-SN_individual_7": 94,
}

ZWAVE_EFFECT_MAPPING = {
    "LZW30-SN": {  # red on/off switch
        "CLEAR": 0,
        "AURORA": 4,
        "MEDIUM_BLINK": 3,
        "CHASE": 2,
        "FAST_CHASE": 2,
        "SLOW_CHASE": 3,
        "FAST_FALLING": 2,
        "MEDIUM_FALLING": 3,
        "SLOW_FALLING": 3,
        "OPEN_CLOSE": 4,
        "FAST_RISING": 2,
        "MEDIUM_RISING": 3,
        "SLOW_RISING": 3,
        "FAST_SIREN": 4,
        "SLOW_SIREN": 4,
        "SMALL_TO_BIG": 4,
    },
    "LZW31-SN": {  # red dimmer
        "AURORA": 4,
        "FAST_BLINK": 3,
        "MEDIUM_BLINK": 4,
        "SLOW_BLINK": 4,
        "CHASE": 2,
        "FAST_CHASE": 2,
        "SLOW_CHASE": 2,
        "FAST_FALLING": 2,
        "MEDIUM_FALLING": 2,
        "SLOW_FALLING": 2,
        "OPEN_CLOSE": 2,
        "PULSE": 5,
        "FAST_RISING": 2,
        "MEDIUM_RISING": 2,
        "SLOW_RISING": 2,
        "SLOW_SIREN": 2,
        "FAST_SIREN": 2,
        "SMALL_TO_BIG": 2,
    },
    "LZW36": {  # red fan/light combo
        "AURORA": 4,
        "FAST_BLINK": 3,
        "MEDIUM_BLINK": 4,
        "SLOW_BLINK": 4,
        "CHASE": 2,
        "FAST_CHASE": 2,
        "SLOW_CHASE": 2,
        "FAST_FALLING": 2,
        "MEDIUM_FALLING": 2,
        "SLOW_FALLING": 2,
        "OPEN_CLOSE": 2,
        "PULSE": 5,
        "FAST_RISING": 2,
        "MEDIUM_RISING": 2,
        "SLOW_RISING": 2,
        "SLOW_SIREN": 2,
        "FAST_SIREN": 2,
        "SMALL_TO_BIG": 2,
    },
}


TRACE: Final = 5
