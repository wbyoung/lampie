"""Fixtures for testing."""

from collections.abc import Generator
import logging
from unittest.mock import AsyncMock, Mock, patch

from freezegun.api import FrozenDateTimeFactory
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
from syrupy.assertion import SnapshotAssertion

from custom_components.lampie.const import (
    CONF_COLOR,
    CONF_EFFECT,
    CONF_SWITCH_ENTITIES,
    DOMAIN,
    TRACE,
)
from custom_components.lampie.types import Integration

from . import MOCK_UTC_NOW, MockNow, add_mock_switch, setup_integration
from .syrupy import LampieSnapshotExtension

_LOGGER = logging.getLogger(__name__)


def pytest_configure(config) -> None:
    is_capturing = config.getoption("capture") != "no"

    if not is_capturing and config.pluginmanager.hasplugin("logging"):
        _LOGGER.warning(
            "pytest run with `-s/--capture=no` and the logging plugin enabled "
            "run with `-p no:logging` to disable all sources of log capturing.",
        )

    # `pytest_homeassistant_custom_component` calls `logging.basicConfig` which
    # creates the `stderr` stream handler. in most cases that will result in
    # logs being duplicated, reported in the "stderr" and "logging" capture
    # sections. force reconfiguration, removing handlers when not running with
    # the `-s/--capture=no` flag.
    if is_capturing:
        logging.basicConfig(level=logging.INFO, handlers=[], force=True)

    logging.getLogger("custom_components.lampie").setLevel(TRACE)
    logging.getLogger("homeassistant").setLevel(logging.INFO)
    logging.getLogger("pytest_homeassistant_custom_component").setLevel(logging.INFO)
    logging.getLogger("asyncio").setLevel(logging.ERROR)


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Enable custom integrations."""
    return


@pytest.fixture(name="mqtt_subscribe", autouse=True)
def auto_patch_mqtt_async_subscribe() -> Generator[AsyncMock]:
    """Patch mqtt.async_subscribe."""

    unsub = Mock()

    with patch(
        "homeassistant.components.mqtt.async_subscribe", return_value=unsub
    ) as mock_subscribe:
        mock_subscribe._unsub = unsub
        yield mock_subscribe


@pytest.fixture
def snapshot(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    """Return snapshot assertion fixture with the Home Assistant extension."""
    return snapshot.use_extension(LampieSnapshotExtension)


@pytest.fixture(name="now")
def mock_now(
    hass: HomeAssistant,
    freezer: FrozenDateTimeFactory,
) -> MockNow:
    """Return a mock now & utcnow datetime."""
    freezer.move_to(MOCK_UTC_NOW)

    return MockNow(hass, freezer)


@pytest.fixture(name="config_entry")
def mock_config_entry() -> MockConfigEntry:
    """Return the default mocked config entry."""
    return MockConfigEntry(
        domain=DOMAIN,
        title="Doors Open",
        entry_id="mock-entry-id",
        data={
            CONF_COLOR: "red",
            CONF_EFFECT: "open_close",
            CONF_SWITCH_ENTITIES: ["light.kitchen"],
        },
    )


@pytest.fixture(name="integration_domain")
def mock_integration_domain() -> Integration:
    """Return the default mocked integration domain."""
    return Integration.ZHA


@pytest.fixture(name="switch")
def mock_switch(
    hass: HomeAssistant, integration_domain: Integration
) -> er.RegistryEntry:
    """Return the default mocked config entry."""
    return add_mock_switch(
        hass,
        "light.kitchen",
        {"manufacturer": "Inovelli", "model": "VZM31-SN"},
        integration=integration_domain,
    )


@pytest.fixture
async def init_integration(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
) -> MockConfigEntry:
    """Set up the Lampie integration for testing."""
    await setup_integration(hass, config_entry)

    return config_entry
