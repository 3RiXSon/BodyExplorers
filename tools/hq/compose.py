"""Resolution-independent UI layer.

render.py draws its title cards, captions and chrome with pixel coordinates
baked for 1280x720. The HQ pipeline renders at 1080p or above, so the overlay
is rebuilt here at native resolution (crisp text, no upscaling) with every
coordinate and font size scaled, plus soft drop shadows for legibility.
"""
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_cache = {}


def font(size, bold=False):
    key = (int(size), bold)
    if key not in _cache:
        _cache[key] = ImageFont.truetype(BOLD if bold else FONT, max(int(size), 6))
    return _cache[key]


def _shadow_text(d, xy, text, fnt, fill, anchor=None, off=2, alpha=150):
    x, y = xy
    d.text((x + off, y + off), text, font=fnt, fill=(0, 0, 0, alpha), anchor=anchor)
    d.text((x, y), text, font=fnt, fill=fill, anchor=anchor)


def overlay(script, caps, i, t, W, H, episode="EP 01", total_scenes=20, shot_len=15.0):
    """Build the RGBA overlay for scene `i` at time `t` (seconds into the shot)."""
    s = W / 1280.0
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    # channel badge
    d.rounded_rectangle((45 * s, 30 * s, 256 * s, 65 * s), radius=16 * s,
                        fill=(18, 62, 76, 240))
    d.text((61 * s, 37 * s), "BODY EXPLORERS", font=font(19 * s, True), fill=(104, 238, 220))
    d.text((W - 146 * s, 39 * s), f"{episode}  /  {i + 1:02}", font=font(20 * s),
           fill=(161, 185, 202))

    # title + teaching label
    _shadow_text(d, (W / 2, 109 * s), script[i]["title"], font(39 * s, True),
                 (255, 250, 233), anchor="mm", off=max(2, 3 * s))
    _shadow_text(d, (W / 2, 155 * s), script[i]["tag"], font(23 * s),
                 (137, 222, 219), anchor="mm", off=max(1, 2 * s))

    # captions
    ci = next((k for k, (a, b, c) in enumerate(caps[i]) if a <= t < b), -1)
    if ci >= 0:
        txt = caps[i][ci][2]
        fnt = font(29 * s)
        lines, line = [], ""
        for word in txt.split():
            test = (line + " " + word).strip()
            if fnt.getlength(test) > 1100 * s:
                lines.append(line)
                line = word
            else:
                line = test
        if line:
            lines.append(line)
        hh = 42 * s * len(lines) + 24 * s
        top = H - 42 * s - hh
        d.rounded_rectangle((60 * s, top, W - 60 * s, H - 36 * s), radius=18 * s,
                            fill=(3, 15, 27, 231))
        for j, ln in enumerate(lines):
            _shadow_text(d, (W / 2, top + 17 * s + j * 42 * s), ln, fnt,
                         (255, 255, 246), anchor="mt", off=max(1, 2 * s))

    # quiz answer labels
    if script[i]["scene"] == "quiz":
        qi = 0 if t < 5.85 else (1 if t < 9.15 else 2)
        labels = [("LUNGS", "STOMACH"), ("HEART", "BONE"), ("RED BLOOD CELLS", "")][qi]
        if qi < 2:
            for x, label in zip([420 * s, 861 * s], labels):
                _shadow_text(d, (x, 526 * s), label, font(26 * s, True),
                             (255, 215, 105), anchor="mm", off=max(2, 2 * s))

    # progress bar
    d.rounded_rectangle((46 * s, H - 18 * s, W - 46 * s, H - 12 * s), radius=3 * s,
                        fill=(31, 61, 79))
    frac = (i + t / shot_len) / float(total_scenes)
    d.rounded_rectangle((46 * s, H - 18 * s, 46 * s + (W - 92 * s) * frac, H - 12 * s),
                        radius=3 * s, fill=(83, 220, 203))
    return im
