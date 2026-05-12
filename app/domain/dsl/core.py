from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

from app.domain.logical_reference import normalize_net_role, normalize_role_label


@dataclass(eq=False)
class Net:
    """Logical net in a DSL reference circuit."""

    name: str
    role: str = "signal"
    label: str | None = None
    circuit: Circuit | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.role = normalize_net_role(self.role)
        if self.label is not None:
            self.label = normalize_role_label(self.label)

    def connect(self, pin: Pin) -> Net:
        pin.connect(self)
        if self.circuit is not None:
            self.circuit._register_component(pin.component)
        return self

    def __and__(self, other: Any) -> Any:
        if isinstance(other, Pin):
            self.connect(other)
            return other
        if isinstance(other, PinSelection):
            if len(other.pins) != 1:
                raise ValueError("& chaining requires a single pin selection")
            self.connect(other.pins[0])
            return other.pins[0]
        raise TypeError(f"cannot connect Net to {type(other).__name__}")


@dataclass(eq=False)
class Pin:
    """Component pin descriptor bound to one component instance."""

    component: Component
    name: str
    role: str | None = None
    net: Net | None = None
    nc: bool = False

    def connect(self, net: Net) -> None:
        self.net = net
        self.nc = False

    def no_connect(self) -> None:
        self.net = None
        self.nc = True

    def __and__(self, other: Any) -> Any:
        if isinstance(other, Net):
            other.connect(self)
            return other
        if isinstance(other, Pin):
            if self.net is None:
                raise ValueError("left pin must already be connected to a net")
            self.net.connect(other)
            return other
        raise TypeError(f"cannot connect Pin to {type(other).__name__}")


class PinSelection:
    """Temporary target returned by Component.__getitem__ for += wiring."""

    def __init__(self, component: Component, pins: Sequence[Pin]):
        self.component = component
        self.pins = list(pins)

    def __iter__(self) -> Iterator[Pin]:
        return iter(self.pins)

    def __iadd__(self, nets: Any) -> PinSelection:
        if isinstance(nets, Net):
            net_items = [nets]
        else:
            net_items = list(nets)
        if len(net_items) != len(self.pins):
            raise ValueError(
                f"{self.component.ref_id} selected {len(self.pins)} pins but got {len(net_items)} nets"
            )
        for pin, net in zip(self.pins, net_items):
            if not isinstance(net, Net):
                raise TypeError(f"expected Net, got {type(net).__name__}")
            net.connect(pin)
        return self

    def __and__(self, other: Any) -> Any:
        if len(self.pins) != 1:
            raise ValueError("& chaining requires a single pin selection")
        return self.pins[0].__and__(other)


class Component:
    """Base class for DSL components."""

    component_type = "Component"
    default_pins: tuple[str, ...] = ("pin1", "pin2")
    default_pin_roles: dict[str, str] = {}

    def __init__(
        self,
        ref_id: str,
        *,
        value: str | None = None,
        description: str | None = None,
        subtype: str | None = None,
        pins: Iterable[str] | None = None,
        **metadata: Any,
    ):
        self.ref_id = str(ref_id)
        self.value = value
        self.description = description
        self.subtype = subtype
        self.metadata = {key: value for key, value in metadata.items() if value is not None}
        pin_names = tuple(pins or self.default_pins)
        self._pins = {
            str(pin_name): Pin(
                component=self,
                name=str(pin_name),
                role=self.default_pin_roles.get(str(pin_name)),
            )
            for pin_name in pin_names
        }

    @property
    def pins(self) -> list[Pin]:
        return list(self._pins.values())

    def pin(self, name: str | int) -> Pin:
        pin_name = self._normalize_pin_name(name)
        if pin_name not in self._pins:
            self._pins[pin_name] = Pin(component=self, name=pin_name)
        return self._pins[pin_name]

    def nc(self, *pins: str | int) -> Component:
        for item in pins:
            self.pin(item).no_connect()
        return self

    def __getitem__(self, key: Any) -> PinSelection:
        if isinstance(key, tuple):
            pins = [self.pin(item) for item in key]
        else:
            pins = [self.pin(key)]
        return PinSelection(self, pins)

    def __setitem__(self, key: Any, value: Any) -> None:
        # Python calls __setitem__ after augmented assignment on a subscript.
        # The wiring side effect already happened in PinSelection.__iadd__.
        return None

    def _normalize_pin_name(self, value: str | int) -> str:
        if isinstance(value, int):
            if value < 1:
                raise ValueError("pin indexes are 1-based")
            defaults = list(self.default_pins)
            if value <= len(defaults):
                return defaults[value - 1]
            return f"pin{value}"
        return str(value)


class Circuit:
    """Container for a Python DSL logical reference circuit."""

    def __init__(
        self,
        *,
        reference_id: str,
        name: str = "",
        description: str = "",
        created_at: str | None = None,
        source: dict[str, Any] | None = None,
        compare_options: dict[str, Any] | None = None,
        **metadata: Any,
    ):
        self.reference_id = reference_id
        self.name = name
        self.description = description
        self.created_at = created_at
        self.source = dict(source or {})
        self.compare_options = dict(compare_options or {})
        self.metadata = {key: value for key, value in metadata.items() if value is not None}
        self._nets: dict[str, Net] = {}
        self._components: dict[str, Component] = {}
        self._symmetry_groups: list[dict[str, Any]] = []

    @property
    def nets(self) -> list[Net]:
        return list(self._nets.values())

    @property
    def components(self) -> list[Component]:
        return list(self._components.values())

    @property
    def symmetry_groups(self) -> list[dict[str, Any]]:
        return list(self._symmetry_groups)

    def net(self, name: str, *, role: str = "signal", label: str | None = None, **metadata: Any) -> Net:
        net_name = str(name)
        existing = self._nets.get(net_name)
        if existing is not None:
            existing.role = normalize_net_role(role or existing.role)
            if label is not None:
                existing.label = normalize_role_label(label)
            existing.metadata.update({key: value for key, value in metadata.items() if value is not None})
            return existing
        net = Net(net_name, role=role, label=label, circuit=self, metadata=metadata)
        self._nets[net_name] = net
        return net

    def input(self, name: str, *, label: str | None = None, **metadata: Any) -> Net:
        return self.net(name, role="input", label=label, **metadata)

    def output(self, name: str, *, label: str | None = None, **metadata: Any) -> Net:
        return self.net(name, role="output", label=label, **metadata)

    def power(self, name: str = "VCC", *, label: str | None = None, **metadata: Any) -> Net:
        return self.net(name, role="power", label=label, **metadata)

    def ground(self, name: str = "GND", *, label: str | None = None, **metadata: Any) -> Net:
        return self.net(name, role="ground", label=label, **metadata)

    def add(self, *components: Component) -> Circuit:
        for component in components:
            self._register_component(component)
        return self

    def symmetry(self, *nets: Net | str, mode: str = "swap_allowed") -> Circuit:
        labels = [net.name if isinstance(net, Net) else str(net) for net in nets]
        if len(labels) < 2:
            raise ValueError("symmetry requires at least two nets")
        self._symmetry_groups.append({"mode": mode, "nets": [labels]})
        return self

    def to_logical_reference(self) -> dict[str, Any]:
        from app.domain.dsl.compile import circuit_to_logical_reference

        return circuit_to_logical_reference(self)

    def _register_component(self, component: Component) -> None:
        existing = self._components.get(component.ref_id)
        if existing is not None and existing is not component:
            raise ValueError(f"duplicate component ref_id: {component.ref_id}")
        self._components[component.ref_id] = component
