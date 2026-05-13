# Lampie for Home Assistant

[![HACS](https://img.shields.io/badge/default-grey?logo=homeassistantcommunitystore&logoColor=white)][hacs-repo]
[![HACS installs](https://img.shields.io/github/downloads/wbyoung/lampie/latest/total?label=installs&color=blue)][hacs-repo]
[![Version](https://img.shields.io/github/v/release/wbyoung/lampie)][releases]
![Downloads](https://img.shields.io/github/downloads/wbyoung/lampie/total)
![Build](https://img.shields.io/github/actions/workflow/status/wbyoung/lampie/pytest.yml)
[![Github Sponsors](https://img.shields.io/badge/GitHub%20Sponsors-grey?&logo=GitHub-Sponsors&logoColor=EA4AAA)][gh-sponsors]

Orchestrate notifications across multiple Inovelli switches.

What it can do:

- **Multiple switches**  
  Display a single LED effect across several physical switches.
- **Shared dismissal**  
  Dismissal of the notification from any switch will dismiss the notification on all switches.
- **Notification priority**  
  Display notifications with a predefined priority so that the most important is shown first.
- **Customizable timeframes**  
  Notifications can have durations of any length of time & are not constrained by what the switch firmware supports, i.e. 90 minutes.
- **State monitoring**  
  Entities are created allowing you to tell when notifications are displayed and details about color, effect, duration, etc.
- **Customized actions**  
  A custom script can be run & modify behavior when a notification starts or ends.
- **Even more**  
  There are some more goodies in to docs below. Enjoy!

_Note: currently this is limited to Blue switches using ZHA or Zigbee2MQTT and Red switches using Z-Wave JS, but other Inovelli switches will be added in the future._

**Configure notifications** for multiple switches easily:

<img width="300" alt="Image" src="https://github.com/user-attachments/assets/02f4888b-836c-4114-8a1d-bff66738087e" />

The integration creates a **simple switch** to turn the notification on and off:

<img width="300" alt="Image" src="https://github.com/user-attachments/assets/1ed2590f-7fc3-4ff9-99d2-a50cdf75a6c1" />

And sensors on each of the switches to see the **current state of each switch**:

<img width="300" alt="Image" src="https://github.com/user-attachments/assets/fc29de87-be5c-47e7-ad35-e1cd126a45fa" />

_Because sometimes one notification may have a [higher priority](#notification-priority) and override another._

## Installation

### HACS

Installation through [HACS][hacs] is the preferred installation method.

[![Open the Lampie integration in HACS][hacs-badge]][hacs-open]

1. Click the button above or go to HACS &rarr; Integrations &rarr; search for
   "Lampie" &rarr; select it.
1. Press _DOWNLOAD_.
1. Select the version (it will auto select the latest) &rarr; press _DOWNLOAD_.
1. Restart Home Assistant then continue to [the setup section](#setup).

### Manual Download

1. Go to the [release page][releases] and download the `lampie.zip` attached
   to the latest release.
1. Unpack the zip file and move `custom_components/lampie` to the following
   directory of your Home Assistant configuration: `/config/custom_components/`.
1. Restart Home Assistant then continue to [the setup section](#setup).

## Setup

Open your Home Assistant instance and start setting up by following these steps:

1. Navigate to "Settings" &rarr; "Devices & Services"
1. Click "+ Add Integration"
1. Search for and select &rarr; "Lampie"

Or you can use the My Home Assistant Button below.

[![Add Integration](https://my.home-assistant.io/badges/config_flow_start.svg)][config-flow-start]

Follow the instructions to configure the integration.

### Configuration Settings

- Choose a name for the notification.
- Choose a color for the notification. The color can be one of the following predefined values or a number in the range 0 - 255:
  - `red` &rarr; 0
  - `blue` &rarr; 170
  - `cyan` &rarr; 130
  - `green` &rarr; 90
  - `pink` &rarr; 230
  - `orange` &rarr; 25
  - `yellow` &rarr; 45
  - `purple` &rarr; 200
  - `white` &rarr; 255
- Choose the effect type.
- Choose a duration in seconds. If left empty, the notification will stay active until dismissed.
- If desired, configure [advanced configuration](#advanced-configuration-settings) options.
- When you proceed, you may need to choose [priorities for switches](#notification-priority).

#### Notification Priority

When a switch is used for multiple notifications, you may need to choose a priority with which to display notifications. For instance, if you have configured a notification _Doors Unlocked_ to be presented on `light.kitchen`, and are now adding a new notification, _Medicine Reminder_, you'll have to choose which will be displayed if both notifications are active. You'll be prompted to provide a list of slugs. In this case, if you wanted the medicine reminder displayed when both are active, you would provide:

```yaml
- medicine_reminder
- doors_unlocked
```

### Advanced Configuration Settings

The following advanced configuration options are available:

#### Start Action

A script that will be called when the notification is activated. This can occur when the notification first turns on or when the notification is re-activated due to an service/action call. The following fields & response variables are used:

- Input fields:
  - `notification`: The slug of the notification being activated.
  - `leds`: The current configuration of the LEDs for the notification. If being activated, this is how they're configured by default. If the notification is active and being re-activated, this is the current LED configuration (respecting any override used in [`lampie.activate`](#lampieactivate)).
- Response variables _all are optional_:
  - `leds` _default_ &raquo; `null`: An override LED configuration to use or `null` to use the already configured value. See: [`examples/start-action.yaml`](examples/start-action.yaml).
  - `block_activation` _default_ &raquo; `False`: Allow blocking of activation (including marking the notification as `on`).

#### End Action

A script that will be called when the notification is deactivated. This can occur because the notification was turned off via the [`switch.<notification_name>_notification` entity](#switchnotification_name_notification) being toggled off, the duration of the notification expiring, or being dismissed on the physical switch. The following fields & response variables are used:

- Input fields:
  - `notification`: The slug of the notification that is ending.
  - `switch_id`: If dismissed on the physical switch, the entity ID of the switch.
  - `device_id`: If dismissed on the physical switch, the device ID of the switch.
  - `dismissed`: True if being dismissed on the physical switch.
- Response variables _all are optional_:
  - `block_dismissal` _default_ &raquo; `False`: Block the dismissal of this notification. This is only considered when the notification is dismissed on the physical switch and the input field `dismissed` is true.
  - `block_next` _default_ &raquo; `False`: Block the immediate re-display of another notification on all switches made available by this notification ending. This can be useful if you know a switch is going to be used for something else immediately after the notification ends.

#### Full LED Configuration

Individual LEDs can be configured via this option. The configuration should provide a list of items for each LED of your switch (starting at the bottom) with the following keys:

- `color` _default_ &raquo; `blue`: The predefined color (see above) or number in the range 0 - 255.
- `effect` _default_ &raquo; `solid`: The effect type.
- `brightness` _default_ &raquo; `100`: The brightness percentage in the range 0 - 100.
- `duration` _default_ &raquo; `null`: The duration in seconds or `null` for a notification that does not expire.

## Entities

Several entities are created for for each notification across the _sensor_ and _switch_ platforms. Some are created for the specific notification while others are created for each switch that is targeted:

- [`switch.<notification_name>_notification`](#switchnotification_name_notification)
- [`sensor.<switch_id>_notification`](#sensorswitch_id_notification)
- [`sensor.<switch_id>_brightness`](#sensorswitch_id_brightness)
- [`sensor.<switch_id>_color`](#sensorswitch_id_color)
- [`sensor.<switch_id>_duration`](#sensorswitch_id_duration)
- [`sensor.<switch_id>_effect`](#sensorswitch_id_effect)

### `switch.<notification_name>_notification`

The switch to turn on or off the notification.

#### Attributes

- `started_at`: The time the notification started, only present when a duration is used.
- `expires_at`: The time the notification expires; only present when a duration is used and it's not supported by the switch firmware.

### `sensor.<switch_id>_notification`

The slug of the active notification.

_The slug is based on the title of the config entry that activated the notification._

### `sensor.<switch_id>_brightness`

The brightness of the LEDs for the active notification.

When using [individual LEDs](#full-led-configuration):

- _Sensor value_: Average brightness
- _Added attribute_: `individual` with the brightness of each LED

### `sensor.<switch_id>_color`

The color of the active notification. This may be either a string or a number depending on how it was configured.

When using [individual LEDs](#full-led-configuration):

- _Sensor value_: A color if only one is used, otherwise `multi`
- _Added attribute_: `individual` with the color of each LED

### `sensor.<switch_id>_duration`

The duration of the active notification.

When using [individual LEDs](#full-led-configuration):

- _Sensor value_: Maximum duration
- _Added attribute_: `individual` with the duration of each LED

#### Attributes

- `started_at`: The time the [switch override](#lampieoverride) started, only present when a duration is used.
- `expires_at`: The time the [switch override](#lampieoverride) expires; only present when a duration is used and it's not supported by the switch firmware.

### `sensor.<switch_id>_effect`

The effect type of the active notification.

When using [individual LEDs](#full-led-configuration):

- _Sensor value_: An effect type if only one is used, otherwise `multi`
- _Added attribute_: `individual` with the effect type of each LED

## Actions

### `lampie.activate`

Activate a notification. This can also be used to re-activate a notification that is currently active. The normal sequence of steps including the [start action](#start-action) will be executed.

#### Service Data Attributes

- `notification`: **required** The slug of the notification to display. Example: `doors_unlocked`.
- `leds`: The LED configuration to use. This overrides the value in the config entry as well as any value returned by the start action script. The data structure for this field matches that of the [_Full LED configuration_](#full-led-configuration) described in the [_Advanced Configuration_ section](#advanced-configuration-settings). Providing a single value will issue an effect on all LEDs rather than individual LEDs.

### `lampie.override`

Apply an LED configuration to an individual switch, overriding any notifications that are being displayed or will be displayed.

_Note: it is valid to target any supported switch. It does not need to be part of a configured Lampie notification. If, however, you want the corresponding entities created to be able to monitor the state of the switch, i.e. [`sensor.<switch_id>_color`](#sensorswitch_id_color), you'll need to create a dummy notification configuration. For instance, you could create a configuration called *Miscellaneous* and disable the corresponding `switch.miscellaneous_notification` entity._

#### Service Data Attributes

- `entity_id`: **required** The entity ID(s) on which to display an effect. Example: `light.kitchen`.
- `leds`: The LED configuration to use. This overrides the value in the config entry as well as any value returned by the start action script. The data structure for this field matches that of the [_Full LED configuration_](#full-led-configuration) described in the [_Advanced Configuration_ section](#advanced-configuration-settings). Providing a single value will issue an effect on all LEDs rather than individual LEDs. Using `null` will clear the override.
- `name` _default_ &raquo; `lampie.override`: A name to use that will be used to identify this override and used for the value of [`sensor.<switch_id>_notification`](#sensorswitch_id_notification).

## Events

### `lampie.dismissed`

This event is fired when a notification or override is dismissed due to dismissal from the physical switch.

#### Event Data

For dismissal of notifications:

- `notification`: The slug of the notification being dismissed.

For dismissal of overrides:

- `entity_id`: The switch ID.
- `override`: The name of the override being dismissed (this is the value that was provided when the override action was invoked).

### `lampie.expired`

This event is fired when a notification or override ends due to the duration expired. When using [individual LEDs](#full-led-configuration), it will only fire when all durations have expired.

#### Event Data

For ending of notifications:

- `notification`: The slug of the notification that is ending.

For ending of overrides:

- `entity_id`: The switch ID.
- `override`: The name of the override that is ending (this is the value that was provided when the override action was invoked).

#### Miscellaneous

Restore state functionality is provided via a subset of entities:

- [`switch.<notification_name>_notification`](#switchnotification_name_notification)
- [`sensor.<switch_id>_notification`](#sensorswitch_id_notification)

If you disable these entities, it is possible that various other entities may not be restored after restarting Home Assistant.

##### Z-Wave

Only the Red series switches are supported by this integration as the Black series switches do not have a concept of notifications.

Some of the older Red series switches only have a single LED or have a subset of available effects. Lampie will do the following for these switches:

- If an unsupported effect is used, it will choose something similar
- If [individual LEDs](#full-led-configuration) are used on a switch with just one LED, the first LED settings will be used

Unlike the Blue series switches under ZHA, there is no way to receive events for when a notification expires (it only supports, for instance, when the config button is dobule pressed `property_key_name="003"` and `value="KeyPressed2x"`). This may be supported in the firmware and not yet available for end user consumption.

This integration therefore handles notification expiration itself for switches configured with Z-Wave. This may change unexpectedly in the future—if and when it is possible, Lampie will change to sending durations to the firmware.

## More Screenshots

Once configured, the integration links the various entities to logical devices:

<img width="300" alt="Image" src="https://github.com/user-attachments/assets/b961a8f6-0393-41bf-a9cb-185fd83c45f9" />

## Credits

Icon designed by [Oblak Labs][oblak-labs-attribution].

[oblak-labs-attribution]: https://thenounproject.com/creator/oblaklabs
[config-flow-start]: https://my.home-assistant.io/redirect/config_flow_start/?domain=lampie
[hacs]: https://hacs.xyz/
[hacs-repo]: https://github.com/hacs/integration
[hacs-badge]: https://my.home-assistant.io/badges/hacs_repository.svg
[hacs-open]: https://my.home-assistant.io/redirect/hacs_repository/?owner=wbyoung&repository=lampie&category=integration
[releases]: https://github.com/wbyoung/lampie/releases
[gh-sponsors]: https://github.com/sponsors/wbyoung
