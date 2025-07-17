"""Lampie Orchestrator tests."""

from homeassistant.core import HomeAssistant

from custom_components.lampie.orchestrator import LampieOrchestrator


def test_teardown_without_setup(hass: HomeAssistant):
    orchestrator = LampieOrchestrator(hass)
    orchestrator.teardown()
