"""One-off fetch of OpenStreetMap standard tiles around Muhanga, stitched into the viewer basemap.

Respects the OSM tile usage policy: about 72 tiles, fetched once, sequentially, with an identifying
User-Agent, and cached locally, so rerunning downloads nothing. Run: uv run tools/fetch_tiles.py
"""

import json
import math
import time
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "tools" / "tile_cache"
OUT = ROOT / "environment" / "viewer"

ZOOM = 12
LON = (29.27, 30.06)
LAT = (-2.46, -1.76)
URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
USER_AGENT = "SkyBlood-coursework-viewer/0.1 (one-off basemap fetch of ~72 tiles, cached)"
HUB = (-2.0792548, 29.7803631)
TILE = 256


def tile_x(lon):
    return (lon + 180) / 360 * 2**ZOOM


def tile_y(lat):
    r = math.radians(lat)
    return (1 - math.asinh(math.tan(r)) / math.pi) / 2 * 2**ZOOM


def lon_of(x):
    return x / 2**ZOOM * 360 - 180


def lat_of(y):
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / 2**ZOOM))))


def to_km(lat, lon):
    """Inverse of the viewer's km -> lat/lon conversion (equirectangular about the hub)."""
    return (lon - HUB[1]) * 111.320 * math.cos(math.radians(HUB[0])), (lat - HUB[0]) * 110.574


def fetch(x, y) -> Path:
    path = CACHE / f"{ZOOM}_{x}_{y}.png"
    if not path.exists():
        req = urllib.request.Request(URL.format(z=ZOOM, x=x, y=y), headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as r:
            path.write_bytes(r.read())
        time.sleep(0.5)
    return path


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    xs = range(math.floor(tile_x(LON[0])), math.floor(tile_x(LON[1])) + 1)
    ys = range(math.floor(tile_y(LAT[1])), math.floor(tile_y(LAT[0])) + 1)
    print(f"zoom {ZOOM}: {len(xs)} x {len(ys)} = {len(xs) * len(ys)} tiles")

    img = Image.new("RGB", (len(xs) * TILE, len(ys) * TILE))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            img.paste(Image.open(fetch(x, y)).convert("RGB"), (i * TILE, j * TILE))
    img.save(OUT / "basemap.jpg", quality=85)

    west, east = lon_of(xs[0]), lon_of(xs[-1] + 1)
    north, south = lat_of(ys[0]), lat_of(ys[-1] + 1)
    x_min, y_max = to_km(north, west)
    x_max, y_min = to_km(south, east)
    bounds = {
        "zoom": ZOOM, "west": west, "east": east, "north": north, "south": south,
        "x_min_km": x_min, "x_max_km": x_max, "y_min_km": y_min, "y_max_km": y_max,
        "width_px": img.width, "height_px": img.height,
        "attribution": "© OpenStreetMap contributors",
    }
    (OUT / "basemap.json").write_text(json.dumps(bounds, indent=2))
    print(f"wrote basemap.jpg ({img.width}x{img.height}) and basemap.json: {bounds}")


if __name__ == "__main__":
    main()
