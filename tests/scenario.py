"""Scenario helpers."""

from __future__ import annotations

import functools
import inspect
from itertools import starmap
from typing import Any

from homeassistant.helpers import entity_registry as er
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from . import ABSENT_NONE
from .stages import StageHandler, StageWrapper, staged_test

type ScenarioStageWrapper = StageWrapper


class Scenario:
    """Scenario for repeatable tet parametrization."""

    ABSENT = ABSENT_NONE

    def __init__(self, scenario_id: str, args: dict[str, Any]) -> None:
        super().__init__()
        self._id = scenario_id
        self._args = args

    @classmethod
    def parametrize(cls, *args: Scenario, **kwargs: Any) -> Any:
        scenarios = args + tuple(starmap(Scenario, kwargs.items()))
        ids = []
        argvalues = []
        argnames = tuple(
            {key: None for scenario in scenarios for key in scenario._args}.keys()
        )

        for item in scenarios:
            ids.append(item._id)
            argvalues.append(
                tuple(item._args.get(key, ABSENT_NONE) for key in argnames)
            )

        return pytest.mark.parametrize(argnames=argnames, argvalues=argvalues, ids=ids)


def staged_scenario_test(
    fn: StageHandler | None = None,
    *,
    standard_config_entry: bool | None = None,
    **kwargs: Any,
):
    if fn is not None:
        return staged_test(
            fn,
            extra_parameters=(
                inspect.signature(__standard_config_entry_signature).parameters.values()
                if standard_config_entry
                else []
            ),
            **kwargs,
        )
    return functools.partial(
        staged_scenario_test, standard_config_entry=standard_config_entry, **kwargs
    )


async def __standard_config_entry_signature(
    config_entry: MockConfigEntry,
    switch: er.RegistryEntry,
):
    pass
