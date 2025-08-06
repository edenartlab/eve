from __future__ import annotations
import os
import re
import textwrap
from PIL import ImageFont


def text_to_lines(text):
    pattern = r"^\d+[\.:]\s*\"?"
    lines = [line for line in text.split("\n") if line]
    lines = [re.sub(pattern, "", line, flags=re.MULTILINE) for line in lines]
    return lines


def get_font(font_name, font_size):
    font_path = os.path.join(os.path.dirname(__file__), "..", "fonts", font_name)
    font = ImageFont.truetype(font_path, font_size)
    return font


def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        if draw.textlength(" ".join(current_line + [word]), font=font) > max_width:
            lines.append(" ".join(current_line))
            current_line = [word]
        else:
            current_line.append(word)
    if current_line:
        lines.append(" ".join(current_line))
    return lines


def concat_sentences(*sentences):
    return " ".join([s.strip().rstrip(".") + "." for s in sentences if s and s.strip()])