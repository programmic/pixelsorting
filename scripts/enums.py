from __future__ import annotations

from enum import Enum
from typing import Any
import json
from pathlib import Path


class Slot(Enum):
    SLOT0 = "slot0"
    SLOT1 = "slot1"
    SLOT2 = "slot2"
    SLOT3 = "slot3"
    SLOT4 = "slot4"
    SLOT5 = "slot5"
    SLOT6 = "slot6"
    SLOT7 = "slot7"
    SLOT8 = "slot8"
    SLOT9 = "slot9"
    SLOT10 = "slot10"
    SLOT11 = "slot11"
    SLOT12 = "slot12"
    SLOT13 = "slot13"
    SLOT14 = "slot14"
    SLOT15 = "slot15"

    @classmethod
    def from_value(cls, v: Any) -> "Slot":
        if isinstance(v, cls):
            return v
        if not isinstance(v, str):
            raise ValueError(f"Cannot convert {v!r} to Slot")
        for s in cls:
            if s.value == v:
                return s
        raise ValueError(f"Unsupported Slot: {v}")


class ControlType(Enum):
    IMAGE_INPUT = "image_input"
    SWITCH = "switch"
    SLIDER = "slider"
    DUALSLIDER = "dualslider"
    DROPDOWN = "dropdown"
    RADIO = "radio"
    COLORPICKER = "colorpicker"

    @classmethod
    def from_value(cls, v: Any) -> "ControlType":
        if isinstance(v, cls):
            return v
        if not isinstance(v, str):
            raise ValueError(f"Cannot convert {v!r} to ControlType")
        v_low = v.lower()
        for c in cls:
            if c.value == v_low:
                return c
        raise ValueError(f"Unsupported ControlType: {v}")


def save_enums_to_json(file_path: str, enums: dict[str, list[str]]) -> None:
    """Save enums to a JSON file."""
    with open(file_path, "w") as f:
        json.dump(enums, f, indent=4)


def load_enums_from_json(file_path: str) -> dict[str, list[str]]:
    """Load enums from a JSON file."""
    if not Path(file_path).exists():
        return {}
    with open(file_path, "r") as f:
        return json.load(f)


def generate_enum(name: str, values: list[str]) -> type[Enum]:
    """Dynamically generate an Enum class."""
    return Enum(name, {value.upper(): value for value in values})


# Example usage:
# enums = {
#     "DynamicEnum": ["value1", "value2", "value3"]
# }
# save_enums_to_json("enums.json", enums)
# loaded_enums = load_enums_from_json("enums.json")
# DynamicEnum = generate_enum("DynamicEnum", loaded_enums["DynamicEnum"])


# def testFunction(img: Image.Image, # Image is always the first parameter, and is not listed in the settings as it is required and handled separately.
#                  # if a secon image is added, it will also be handled separately and not listed in the settings.
#                  # should there be a third image, or should any image be titled mask or be optional, it should be listed in the settings.
#                  param1: int,
#                  param2: Enum,
#                  param3: Enum,
#                  param4: bool,
#                  param5: float,
#                  param6: tuple[int, int],
#                  param7: tuple[float, float]
#                  ):
#     # not important
#     return None