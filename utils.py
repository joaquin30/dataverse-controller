import colorcet as cc
from PIL import ImageColor

def get_normalized_colors(length: int) -> list[list[float]]:
    return list(map(hex_to_float_rgb, cc.glasbey_light[:length]))

def hex_to_float_rgb(hex: str) -> list[float]:
    color = ImageColor.getcolor(hex, "RGB")
    return list(map(lambda x: x / 256, color))

def get_colors(length: int) -> list[tuple[int, int, int]]:
    return list(map(lambda x: ImageColor.getcolor(x, "RGB"), cc.glasbey_light[:length]))

def invert_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(map(lambda x: 255 - x, color))