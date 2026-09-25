#!/usr/bin/env python3
"""Render social-media demo cards for intent-monitor.

Authentic terminal frames are rendered by **agg** (the asciinema GIF engine),
then composited into a captioned social card (framed window + big Inter caption
pill + a clear STEP n / N indicator so the looping sequence has an obvious start
and end). Aspect ratios: 4:5 and 1:1.

    python demo/render_social.py

Requires: agg on PATH (winget install asciinema.agg) and Pillow. Terminal values
(chain-of-thought, ledger, verdicts, reason) are the tool's REAL output; only the
layout, captions, and pacing are presentational. Illustrative, not a benchmark.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SOCIAL = os.path.join(HERE, "social")
INTER = os.path.join(SOCIAL, "Inter-Bold.ttf")

COLS, ROWS, AGG_FONT = 46, 11, 30
# TokyoNight theme for agg: bg,fg + 16 palette colors
THEME = "1a1b26,c0caf5,15161e,f7768e,9ece6a,e0af68,7aa2f7,bb9af7,7dcfff,a9b1d6,414868,f7768e,9ece6a,e0af68,7aa2f7,bb9af7,7dcfff,c0caf5"

# card palette
BACKDROP = (13, 14, 22)
WIN = (26, 27, 38)
BORDER = (41, 46, 66)
FG = (192, 202, 245)
DIM = (120, 130, 170)
FAINT = (86, 95, 137)
ACCENT = (125, 207, 255)
PILL = (30, 32, 48)
DOT_ON = (125, 207, 255)
DOT_OFF = (48, 54, 78)

RED, YEL, GRN, CYN, DMc = (247, 118, 142), (224, 175, 104), (158, 206, 106), (125, 207, 255), (86, 95, 137)


def a(text, rgb):
    r, g, b = rgb
    return f"\x1b[38;2;{r};{g};{b}m{text}\x1b[0m"


SCENES = [
    {
        "lines": [
            a("$", GRN) + " python demo.py --compact",
            "",
            a("  stated intent (from the CoT)", DMc),
            "  claim: \"" + a("read", CYN) + " the record\"",
            "  claim: \"" + a("update", CYN) + " it as asked\"",
            "",
            a("  action ledger", DMc),
            "  1. read_file  path=record.json",
            "  2. write_file path=record.json",
            a("  3. export  -> sink.example/collect", RED),
        ],
        "caption": "Stated intent: read and update, but the ledger also exports",
        "hold": 3200,
    },
    {
        "lines": [
            "  CoT-only     " + a("NOT FLAGGED", YEL),
            "  action-only  " + a("NOT FLAGGED", YEL),
            "",
            a("  reading words or tool-names", DMc),
            a("  alone misses it", DMc),
        ],
        "caption": "Two common monitors see nothing wrong",
        "hold": 3000,
    },
    {
        "lines": [
            a("  intent -> action licensing", CYN),
            "",
            "  read_file  -> licensed by '" + a("read", CYN) + "'",
            "  write_file -> licensed by '" + a("update", CYN) + "'",
            a("  export     -> UNLICENSED", RED),
        ],
        "caption": "Extract intent from the reasoning, diff it against the actions",
        "hold": 3800,
    },
    {
        "lines": [
            "  ledger-diff   " + a("FLAGGED", RED),
            "",
            a("  unlicensed_egress_action", RED),
            "  export ran, but intent only",
            "  licensed " + a("read", CYN) + " + " + a("update", CYN),
        ],
        "caption": "The one action no stated intent licensed",
        "hold": 3400,
    },
    {
        "lines": [
            a("  intent-monitor", CYN),
            "",
            "  catch agents whose words and",
            "  actions quietly diverge",
        ],
        "caption": "A safety-monitor prototype",
        "sub": "illustrative · deterministic offline demo",
        "hold": 3000,
    },
]

LAYOUT = {
    "4x5": dict(size=(1080, 1350), win_top=168, cap=48, sub=27, label=30),
    "1x1": dict(size=(1080, 1080), win_top=104, cap=44, sub=25, label=28),
}


TYPE_DELAY = 0.045   # seconds per typed character (scene 1 command line)
LINE_DELAY = 0.12    # seconds between revealed output lines
RESET = "\x1b[0m"


def _scene_events(lines, type_first):
    """Build a cast event list that reveals `lines` progressively.

    Scene 1 (type_first) types its first line char-by-char, then reveals the rest
    one line per frame. Other scenes reveal every line one per frame.
    """
    events, t = [], 0.1
    rest_lines = lines
    if type_first and lines:
        line0 = lines[0]
        # keep the colored "$ " prompt, then type the remaining visible command
        cut = line0.index(RESET) + len(RESET) if RESET in line0 else 0
        prompt, typed = line0[:cut], line0[cut:]
        events.append([round(t, 3), "o", prompt])
        t += 0.30
        for ch in typed:
            events.append([round(t, 3), "o", ch])
            t += TYPE_DELAY
        t += 0.20
        events.append([round(t, 3), "o", "\r\n"])
        t += LINE_DELAY
        rest_lines = lines[1:]
    for ln in rest_lines:
        events.append([round(t, 3), "o", ln + "\r\n"])
        t += LINE_DELAY
    events.append([round(t + 0.05, 3), "o", ""])
    return events


def render_scene_frames(lines, type_first, fps_cap):
    """Render one scene's cast via agg and return [(RGB frame, duration_ms), ...].

    ALL agg frames are extracted (not just the last). Each frame is copied out and
    the source GIF is closed before the temp dir is cleaned up (WinError 32 guard).
    """
    header = {"version": 2, "width": COLS, "height": ROWS, "title": "intent-monitor"}
    events = _scene_events(lines, type_first)
    with tempfile.TemporaryDirectory() as td:
        cast = os.path.join(td, "s.cast")
        gif = os.path.join(td, "s.gif")
        with open(cast, "w", encoding="utf-8") as fh:
            fh.write("\n".join([json.dumps(header)] + [json.dumps(e) for e in events]) + "\n")
        subprocess.run(
            ["agg", "--font-family", "Cascadia Mono", "--font-size", str(AGG_FONT),
             "--theme", THEME, "--fps-cap", str(fps_cap), "--last-frame-duration", "0.1",
             cast, gif],
            check=True, capture_output=True,
        )
        im = Image.open(gif)
        frames = []
        for i in range(im.n_frames):
            im.seek(i)
            dur = int(im.info.get("duration", 50) or 50)
            frames.append((im.convert("RGB").copy(), dur))
        im.close()
        return frames


def font(path, size):
    return ImageFont.truetype(path, size)


def wrap(draw, text, fnt, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=fnt) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def build_card_base(scene, idx, total, lo, term_size):
    """Draw everything that is STATIC for a scene (chrome, caption, step, dots).

    Returns (base_image, paste_xy). Each animated terminal frame is pasted into a
    copy of this base at paste_xy, so the caption is present from the first frame.
    """
    W, H = lo["size"]
    img = Image.new("RGB", (W, H), BACKDROP)
    d = ImageDraw.Draw(img)

    tw, th = term_size
    title_h, inner = 58, 22
    win_w = tw + 2 * inner
    win_h = th + title_h + inner
    win_x0 = (W - win_w) // 2
    win_y0 = lo["win_top"]
    win_x1, win_y1 = win_x0 + win_w, win_y0 + win_h

    # STEP n / N label (top-left, above the window)
    lf = font(INTER, lo["label"])
    d.text((win_x0, win_y0 - lo["label"] - 26), f"STEP {idx} / {total}", font=lf, fill=ACCENT)

    # window shadow
    shadow = Image.new("RGB", (W, H), BACKDROP)
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle([win_x0, win_y0 + 14, win_x1, win_y1 + 14], radius=22, fill=(0, 0, 0))
    img = Image.blend(img, shadow.filter(ImageFilter.GaussianBlur(18)), 0.55)
    d = ImageDraw.Draw(img)

    # window + title bar
    d.rounded_rectangle([win_x0, win_y0, win_x1, win_y1], radius=18, fill=WIN, outline=BORDER, width=1)

    # top-right window controls (minimize | maximize | close), faint, evenly spaced
    bar_cy = win_y0 + title_h // 2
    g = 8  # glyph half-extent
    step = 30
    cx_close = win_x1 - 34
    cx_max = cx_close - step
    cx_min = cx_max - step
    d.line([cx_min - g, bar_cy, cx_min + g, bar_cy], fill=FAINT, width=2)                       # minimize
    d.rectangle([cx_max - g, bar_cy - g, cx_max + g, bar_cy + g], outline=FAINT, width=2)        # maximize
    d.line([cx_close - g, bar_cy - g, cx_close + g, bar_cy + g], fill=FAINT, width=2)            # close (\)
    d.line([cx_close - g, bar_cy + g, cx_close + g, bar_cy - g], fill=FAINT, width=2)            # close (/)

    # centered title
    tf = font(INTER, 22)
    tt = "intent-monitor"
    d.text(((W - d.textlength(tt, font=tf)) / 2, win_y0 + (title_h - 26) / 2), tt, font=tf, fill=FAINT)

    # caption pill (lower third)
    capf = font(INTER, lo["cap"])
    px0, px1 = 60, W - 60
    cap_lines = wrap(d, scene["caption"], capf, (px1 - px0) - 80)
    clh = int(lo["cap"] * 1.2)
    sub = scene.get("sub")
    subf = font(INTER, lo["sub"]) if sub else None
    text_h = len(cap_lines) * clh + (int(lo["sub"] * 1.5) if sub else 0)
    pad_y = 32
    pill_h = text_h + 2 * pad_y
    region_top, region_bot = win_y1 + 44, H - 132
    py0 = region_top + max(0, (region_bot - region_top - pill_h) // 2)
    d.rounded_rectangle([px0, py0, px1, py0 + pill_h], radius=24, fill=PILL)
    ty = py0 + pad_y
    for ln in cap_lines:
        d.text(((W - d.textlength(ln, font=capf)) / 2, ty), ln, font=capf, fill=FG)
        ty += clh
    if sub:
        d.text(((W - d.textlength(sub, font=subf)) / 2, ty + 6), sub, font=subf, fill=DIM)

    # progress dots (bottom), current step highlighted
    dot_r, gap = 9, 34
    total_w = (total - 1) * gap + dot_r * 2
    dx = (W - total_w) // 2
    dy = H - 78
    for i in range(total):
        on = i == idx - 1
        d.ellipse([dx, dy - dot_r, dx + dot_r * 2, dy + dot_r], fill=DOT_ON if on else DOT_OFF)
        dx += gap

    return img, (win_x0 + inner, win_y0 + title_h)


def main():
    os.makedirs(SOCIAL, exist_ok=True)
    fps_cap = 20
    hold_ms = 2200

    # terminal frames are aspect-ratio independent, render each scene's cast once
    scene_frames = [
        render_scene_frames(s["lines"], type_first=(i == 0), fps_cap=fps_cap)
        for i, s in enumerate(SCENES)
    ]

    for ratio, lo in LAYOUT.items():
        cards, durs = [], []
        for i, s in enumerate(SCENES):
            frames = scene_frames[i]
            base, paste_xy = build_card_base(s, i + 1, len(SCENES), lo, frames[0][0].size)
            last = None
            for term_img, dur in frames:
                card = base.copy()
                card.paste(term_img, paste_xy)
                cards.append(card)
                durs.append(dur)
                last = card
            cards.append(last)          # hold on the fully-revealed frame
            durs.append(hold_ms)
        out = os.path.join(SOCIAL, f"intent-monitor-{ratio}.gif")
        cards[0].save(out, save_all=True, append_images=cards[1:], duration=durs,
                      loop=0, disposal=2, optimize=True)
        print(f"wrote {out}  ({len(cards)} frames, {os.path.getsize(out) // 1024} KB, "
              f"{lo['size'][0]}x{lo['size'][1]})")


if __name__ == "__main__":
    main()
