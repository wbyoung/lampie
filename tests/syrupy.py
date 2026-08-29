"""Customizations for Syrupy."""

from dataclasses import asdict, fields, replace
from typing import Any

from homeassistant.core import Event, ServiceCall, State
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.util.read_only_dict import ReadOnlyDict
from pytest_homeassistant_custom_component.syrupy import (
    ANY,
    HomeAssistantSnapshotExtension,
    HomeAssistantSnapshotSerializer,
)
from syrupy.extensions.amber import AmberDataSerializer
from syrupy.filters import props
from syrupy.matchers import path_type
from syrupy.types import (
    PropertyFilter,
    PropertyMatcher,
    PropertyName,
    PropertyPath,
    SerializableData,
)

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

        if isinstance(data, er.RegistryEntry):
            base_exclude = exclude
            exclude_props = props(
                # compat for HA DeviceRegistryEntrySnapshot <2025.9.0 and >=2026.2.0
                "object_id_base",
            )

            def combined_exclude(*, prop: PropertyName, path: PropertyPath) -> bool:
                if base_exclude and base_exclude(prop=prop, path=path):
                    return True
                return bool(exclude_props(prop=prop, path=path))

            exclude = combined_exclude

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
    def _serializable_state(cls, data: State) -> SerializableData:
        result = super()._serializable_state(data)
        if "attributes" in result:
            result["attributes"] = ReadOnlyDict(
                {str(key): value for key, value in result["attributes"].items()}
            )
        return result

    @classmethod
    def _serializable_entity_registry_entry(
        cls, data: er.RegistryEntry
    ) -> SerializableData:
        result = super()._serializable_entity_registry_entry(data)
        if "aliases" in result:
            aliases = [alias for alias in result["aliases"] if alias is not None]
            result["aliases"] = set(aliases)
            assert len(aliases) == len(result["aliases"])
        if result.get("capabilities", None):
            result["capabilities"] = {
                str(key): value for key, value in result["capabilities"].items()
            }
        return result

    @classmethod
    def _serializable_device_registry_entry(
        cls, data: dr.DeviceEntry
    ) -> SerializableData:
        result = super()._serializable_device_registry_entry(data)
        if "config_entry_id" in result:
            result["primary_config_entry"] = result.pop(
                "config_entry_id"  # should be ANY
            )
            result["config_entries"] = result["primary_config_entry"]
        if "config_subentry_id" in result:
            result["config_entries_subentries"] = result.pop(
                "config_subentry_id"  # should be ANY
            )
        return result

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
    mapping={r".*\.device_id": (str, None.__class__)},
    replacer=lambda result, _: ANY if result is not None else None,
    regex=True,
)
