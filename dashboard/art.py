import base64
import io
import os
from functools import lru_cache
from pathlib import Path

ASSETS_ENV = "MLPLAT_ASSETS_DIR"
VENDOR_DIR = Path(__file__).resolve().parent / "assets" / "vendor"
PACK_NAME = "Cozy Bakery & Food Icons Pack"
PACK_ARTIST = "Jimal"
PACK_URL = "https://jimal-art.itch.io/bakery-food-asset-pack-vector"

KITCHEN = ("oven_closed", "oven_half_open", "oven_open", "oven_door", "rolling_pin", "flour_sack", "egg_basket",
           "milk_bottle", "baker_peel", "oven_mitt", "golden_coin", "silver_coin")
FOOD = ("01_croissant", "02_pancakes", "03_pretzel", "04_baguette", "05_donut", "06_cinnabon", "07_challah",
        "08_loaf", "09_muffin", "10_waffles", "11_brioche", "12_bread")
STATION_ART = ("rolling_pin", "flour_sack", "egg_basket", "milk_bottle")
DISPLAY_PX = {"oven_closed": 256, "oven_half_open": 256, "oven_open": 256, "oven_door": 256}
DEFAULT_PX = 128


def vendor_dir():
    return Path(os.environ.get(ASSETS_ENV) or VENDOR_DIR)


def file_for(name):
    suffix = "_512.png" if name in FOOD else ".png"
    return vendor_dir() / f"{name}{suffix}"


def available():
    return all(file_for(name).exists() for name in KITCHEN + FOOD)


@lru_cache(maxsize=None)
def encoded(path_text, size):
    from PIL import Image

    with Image.open(path_text) as image:
        image = image.convert("RGBA")
        if image.width > size:
            image = image.resize((size, round(image.height * size / image.width)), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def data_uri(name):
    path = file_for(name)
    if not path.exists():
        return None
    return encoded(str(path), DISPLAY_PX.get(name, DEFAULT_PX))


def food(index):
    return data_uri(FOOD[index % len(FOOD)])


def img(name, alt, css_class="", size=None):
    uri = data_uri(name)
    style = f' style="width:{size}px;height:{size}px"' if size else ""
    if uri is None:
        label = alt.split()[0] if alt else "?"
        return f'<span class="art-missing {css_class}" role="img" aria-label="{alt}"{style}>{label}</span>'
    return f'<img class="art {css_class}" src="{uri}" alt="{alt}"{style}>'


def missing_message():
    return (f"Bakery art not found. Download the {PACK_NAME} by {PACK_ARTIST} from {PACK_URL} and unzip the PNG files "
            f"into dashboard/assets/vendor/. The dashboard works without it.")


def attribution_html():
    return (f'Bakery art: <a href="{PACK_URL}" target="_blank" rel="noopener">{PACK_NAME}</a> by {PACK_ARTIST}. '
            f'Used with attribution; not redistributed.')
