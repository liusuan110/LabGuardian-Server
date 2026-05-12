from __future__ import annotations

from app.domain.dsl.core import Component


class Resistor(Component):
    component_type = "Resistor"
    default_pins = ("pin1", "pin2")


class Capacitor(Component):
    component_type = "Capacitor"
    default_pins = ("pin1", "pin2")


class CapacitorCeramic(Component):
    component_type = "CapacitorCeramic"
    default_pins = ("pin1", "pin2")


class CapacitorElectrolytic(Component):
    component_type = "CapacitorElectrolytic"
    default_pins = ("positive", "negative")
    default_pin_roles = {"positive": "positive", "negative": "negative"}


class LED(Component):
    component_type = "LED"
    default_pins = ("anode", "cathode")
    default_pin_roles = {"anode": "anode", "cathode": "cathode"}


class Diode(Component):
    component_type = "Diode"
    default_pins = ("anode", "cathode")
    default_pin_roles = {"anode": "anode", "cathode": "cathode"}


class Transistor(Component):
    component_type = "Transistor"
    default_pins = ("collector", "base", "emitter")
    default_pin_roles = {
        "collector": "collector",
        "base": "base",
        "emitter": "emitter",
    }


class Potentiometer(Component):
    component_type = "Potentiometer"
    default_pins = ("terminal_a", "wiper", "terminal_b")
    default_pin_roles = {
        "terminal_a": "terminal_a",
        "wiper": "wiper",
        "terminal_b": "terminal_b",
    }


class Wire(Component):
    component_type = "Wire"
    default_pins = ("pin1", "pin2")


class IC(Component):
    component_type = "IC"
    default_pins = ()
