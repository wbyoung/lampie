"""Customizations for Syrupy."""

from dataclasses import asdict, fields, replace
from typing import Any

from homeassistant.core import Event, ServiceCall
from pytest_homeassistant_custom_component.syrupy import (
    ANY,
    HomeAssistantSnapshotExtension,
    HomeAssistantSnapshotSerializer,
)
from syrupy.extensions.amber import AmberDataSerializer
from syrupy.matchers import path_type
from syrupy.types import PropertyFilter, PropertyMatcher, PropertyPath, SerializableData

from custom_components.lampie.types import (
    ExpirationInfo,
    LampieNotificationInfo,
    LampieSwitchInfo,
)


class LampieSnapshotSerializer(HomeAssistantSnapshotSerializer):
    @classmethod
    def _serialize(
        cls,
        data: SerializableData,
        *,
        depth: int = 0,
        exclude: PropertyFilter | None = None,
        include: PropertyFilter | None = None,
        matcher: PropertyMatcher | None = None,
        path: PropertyPath = (),
        visited: set[Any] | None = None,
    ) -> str:
        if isinstance(data, Event):
            serializable_data = cls._serializable_event(data)
        elif isinstance(data, ServiceCall):
            serializable_data = cls._serializable_service_call(data)
        elif isinstance(data, ExpirationInfo):
            serializable_data = cls._serializable_expiration_info(data)
        elif isinstance(data, (LampieNotificationInfo, LampieSwitchInfo)):
            serializable_data = {
                field.name: getattr(data, field.name) for field in fields(data)
            }
        else:
            serializable_data = data

        serialized: str = super()._serialize(
            serializable_data,
            depth=depth,
            exclude=exclude,
            include=include,
            matcher=matcher,
            path=path,
            visited=visited,
        )

        return serialized

    @classmethod
    def _serializable_event(cls, data: Event) -> SerializableData:
        """Prepare a Home Assistant event for serialization."""
        return EventSnapshot(
            data.as_dict() | {"id": ANY, "time_fired": ANY, "context": ANY},
        )

    @classmethod
    def _serializable_service_call(cls, call: ServiceCall) -> SerializableData:
        """Prepare a Home Assistant service call for serialization."""

        return ServiceCallSnapshot(
            {key: getattr(call, key) for key in call.__slots__}
            | {"context": ANY, "hass": ANY},
        )

    @classmethod
    def _serializable_expiration_info(cls, info: ExpirationInfo) -> SerializableData:
        """Prepare Lampie expiration info for serialization."""
        result = {}

        if info.cancel_listener:
            result["cancel_listener"] = ANY
            info = replace(info, cancel_listener=None)

        result = asdict(info) | result

        return ExpirationInfoSnapshot(result)


class LampieSnapshotExtension(HomeAssistantSnapshotExtension):
    serializer_class: type[AmberDataSerializer] = LampieSnapshotSerializer


class EventSnapshot(dict):  # noqa: FURB189
    """Tiny wrapper to represent an event in snapshots."""


class ServiceCallSnapshot(dict):  # noqa: FURB189
    """Tiny wrapper to represent a service call in snapshots."""


class ExpirationInfoSnapshot(dict):  # noqa: FURB189
    """Tiny wrapper to represent expiration info in snapshots."""


any_device_id_matcher = path_type(
    mapping={r".*\.device_id": (str,)},
    replacer=lambda *_: ANY,
    regex=True,
)
