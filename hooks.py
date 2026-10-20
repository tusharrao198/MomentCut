import os
import re
import textwrap
import subprocess
import urllib.request
import uuid
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from ffmpeg_utils import (QUALITY, METADATA_SCRUB, finished_before_crash,
                          run_encode)
import gpu_decode


def _truncate_bytes(text, max_bytes):
    """Trim to a byte budget without splitting a multi-byte character.

    Deliberately duplicated from main.truncate_bytes: this module stays free of
    main's heavy imports (cv2, mediapipe, torch) so it can be used standalone.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", "ignore")

FONT_URL = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSerif/NotoSerif-Bold.ttf"
FONT_DIR = "fonts"
FONT_PATH = os.path.join(FONT_DIR, "NotoSerif-Bold.ttf")
# Bundled (SIL OFL, see fonts/Montserrat-OFL.txt): the "pill" look's sans.
PILL_FONT_PATH = os.path.join(FONT_DIR, "Montserrat-ExtraBold.ttf")

# Typefaces the hook editor offers, all bundled so the server render and the
# browser preview use the same file. The size factor (share of the box width)
# evens out their visual weight: Anton is condensed, Montserrat wide.
HOOK_FONTS = {
    "montserrat": (PILL_FONT_PATH, 0.064),
    "anton": (os.path.join(FONT_DIR, "Anton-Regular.ttf"), 0.08),
    "serif": (FONT_PATH, 0.05),
}

# Codepoint ranges NotoSerif has no glyphs for (would render as tofu boxes).
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # emoticons, symbols, transport, supplemental
    "\U00002600-\U000027BF"  # misc symbols + dingbats
    "\U0001F1E6-\U0001F1FF"  # regional indicators (flags)
    "\U00002B00-\U00002BFF"  # arrows, stars
    "\U0000FE0E\U0000FE0F"   # variation selectors
    "\U0000200D"             # zero-width joiner
    "\U000020E3"             # combining keycap
    "]+"
)

# Emoji-capable fonts, probed at runtime (Windows, WSL, Linux/Docker, macOS).
_EMOJI_FONT_CANDIDATES = [
    "C:\\Windows\\Fonts\\seguiemj.ttf",
    "/mnt/c/Windows/Fonts/seguiemj.ttf",
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto-emoji/NotoColorEmoji.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",
]


# Bitmap strike sizes color-emoji fonts ship with. NotoColorEmoji (the font
# the Docker image installs) ONLY loads at its strike size — asking for an
# arbitrary size raises "invalid pixel size" — so glyphs are rendered at the
# native size and rescaled at draw time.
_EMOJI_BITMAP_SIZES = [109, 128, 136, 160, 96, 72, 64, 32]


def _load_emoji_font(font_size):
    """Return (font, native_size) for an emoji-capable font, or None.

    native_size == font_size for scalable fonts (Segoe UI Emoji); for
    fixed-bitmap fonts (NotoColorEmoji) it is the strike size the font
    actually loaded at, and the renderer scales glyphs down from it."""
    for path in _EMOJI_FONT_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            return ImageFont.truetype(path, font_size), font_size
        except Exception:
            pass
        for native in _EMOJI_BITMAP_SIZES:
            try:
                return ImageFont.truetype(path, native), native
            except Exception:
                continue
    return None


def _split_emoji_runs(text):
    """Split text into (is_emoji, chunk) runs."""
    runs = []
    pos = 0
    for m in _EMOJI_RE.finditer(text):
        if m.start() > pos:
            runs.append((False, text[pos:m.start()]))
        runs.append((True, m.group()))
        pos = m.end()
    if pos < len(text):
        runs.append((False, text[pos:]))
    return runs


def _emoji_scale(font, emoji_font):
    """Draw-time scale factor from the emoji font's native size to the text
    size. 1.0 for scalable emoji fonts; < 1 for fixed-bitmap strikes."""
    efont, native = emoji_font
    target = getattr(font, "size", native)
    return target / float(native) if native else 1.0


def _measure_width(draw, text, font, emoji_font):
    """Pixel width of a line, measuring emoji runs with the emoji font.

    emoji_font is the (font, native_size) pair from _load_emoji_font, or None."""
    width = 0.0
    for is_emoji, chunk in _split_emoji_runs(text):
        if is_emoji and emoji_font:
            width += (draw.textlength(chunk, font=emoji_font[0])
                      * _emoji_scale(font, emoji_font))
        else:
            width += draw.textlength(chunk, font=font)
    return width


def _render_emoji_chunk(chunk, emoji_font, scale, fill):
    """Rasterize an emoji run at the font's native size, scaled to the text
    size. Returns an RGBA image ready to alpha-composite, or None."""
    efont, native = emoji_font
    probe = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
    w = int(probe.textlength(chunk, font=efont))
    if w <= 0:
        return None
    # Emoji glyphs can overshoot the em box slightly; 1.3x covers it.
    tmp = Image.new('RGBA', (w, int(native * 1.3)), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    try:
        d.text((0, 0), chunk, font=efont, embedded_color=True)
    except TypeError:
        d.text((0, 0), chunk, font=efont, fill=fill)
    if scale == 1.0:
        return tmp
    return tmp.resize((max(int(w * scale), 1), max(int(tmp.height * scale), 1)),
                      Image.LANCZOS)


def _draw_mixed(img, draw, xy, text, font, emoji_font, fill, outline=None):
    """Draw a line onto img, rendering emoji runs with the emoji font (in
    color if supported, rescaled when the font is a fixed-size bitmap).
    outline: optional (color, px) stroke drawn under the text."""
    x, y = xy
    stroke_w = outline[1] if outline else 0
    stroke_fill = outline[0] if outline else None
    for is_emoji, chunk in _split_emoji_runs(text):
        if is_emoji and emoji_font:
            scale = _emoji_scale(font, emoji_font)
            rendered = _render_emoji_chunk(chunk, emoji_font, scale, fill)
            if rendered is not None:
                img.alpha_composite(rendered, (int(x), int(y)))
                x += draw.textlength(chunk, font=emoji_font[0]) * scale
        else:
            if stroke_w:
                draw.text((x, y), chunk, font=font, fill=fill,
                          stroke_width=stroke_w, stroke_fill=stroke_fill)
            else:
                draw.text((x, y), chunk, font=font, fill=fill)
            x += draw.textlength(chunk, font=font)


def _break_long_word(draw, word, font, emoji_font, max_width):
    """Character-level hard wrap for a single word wider than max_width."""
    pieces = []
    current = ""
    for ch in word:
        if current and _measure_width(draw, current + ch, font, emoji_font) > max_width:
            pieces.append(current)
            current = ch
        else:
            current += ch
    if current:
        pieces.append(current)
    return pieces

def download_font_if_needed():
    """Downloads a serif font for the hook text if not present."""
    if not os.path.exists(FONT_DIR):
        os.makedirs(FONT_DIR)
    if not os.path.exists(FONT_PATH):
        print(f"⬇️ Downloading font from {FONT_URL}...")
        try:
            # Add user agent to avoid 403s slightly
            req = urllib.request.Request(
                FONT_URL, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req) as response, open(FONT_PATH, 'wb') as out_file:
                out_file.write(response.read())
            print("✅ Font downloaded.")
        except Exception as e:
            print(f"❌ Failed to download font: {e}")

# Hook visual styles. ``pill`` remains the internal name of the default for
# backwards compatibility with saved dashboard preferences and API callers,
# but its visual treatment is now the modern accent look rather than white
# boxes around black text.
HOOK_STYLES = {
    # Default. Premium creator look: bold white Montserrat, the final impact
    # word in warm gold, a thin dark stroke, and one translucent charcoal
    # backdrop. The legacy key is intentionally retained so existing "pill"
    # preferences receive the new default without a migration.
    "pill":    {"box": (12, 15, 20, 165), "text": (255, 255, 255),
                "highlight": (255, 201, 40), "outline": ((0, 0, 0), 4),
                "shadow": True, "font": PILL_FONT_PATH, "accent": True},
    # White card, black serif text (original look).
    "classic": {"box": (255, 255, 255, 240), "text": (0, 0, 0), "outline": None, "shadow": True},
    # Dark card, white text.
    "dark":    {"box": (18, 18, 20, 235),    "text": (255, 255, 255), "outline": None, "shadow": True},
    # Bright yellow card, black text (high-contrast TikTok look).
    "yellow":  {"box": (255, 214, 0, 245),   "text": (0, 0, 0), "outline": None, "shadow": True},
    # Red "breaking" card, white text.
    "red":     {"box": (220, 38, 38, 245),   "text": (255, 255, 255), "outline": None, "shadow": True},
    # No box: white text with a thick black outline (caption/MrBeast style).
    "outline": {"box": (0, 0, 0, 0),         "text": (255, 255, 255), "outline": ((0, 0, 0), 8), "shadow": False},
    # No box: yellow text with black outline.
    "outline_yellow": {"box": (0, 0, 0, 0),  "text": (255, 214, 0),   "outline": ((0, 0, 0), 8), "shadow": False},
}


def _impact_word_span(text):
    """Character span of the final readable word, ignoring trailing emoji.

    Hooks commonly end in punctuation or an emoji. Highlighting the last
    whitespace token would colour that emoji instead of the phrase's payoff,
    so select the final Unicode letter/number word.
    """
    matches = list(re.finditer(r"[^\W_]+(?:['’][^\W_]+)*", text, re.UNICODE))
    if not matches:
        return None
    match = matches[-1]
    return match.start(), match.end()


def _balanced_accent_lines(text, draw, font, emoji_font, max_width):
    """Return one or two balanced lines for the default hook treatment."""
    words = text.replace("\n", " ").split()
    if not words:
        return [" "]
    whole = " ".join(words)
    if _measure_width(draw, whole, font, emoji_font) <= max_width:
        return [whole]
    if len(words) == 1:
        return [whole]

    choices = []
    for split in range(1, len(words)):
        first = " ".join(words[:split])
        second = " ".join(words[split:])
        first_w = _measure_width(draw, first, font, emoji_font)
        second_w = _measure_width(draw, second, font, emoji_font)
        overflow = max(0.0, first_w - max_width) + max(0.0, second_w - max_width)
        choices.append((overflow, abs(first_w - second_w), max(first_w, second_w),
                        [first, second]))
    return min(choices, key=lambda item: item[:3])[3]


def _draw_accent_line(img, draw, x, y, line, font, emoji_font, impact_word,
                      highlight_this_line, text_fill, highlight_fill, outline):
    """Draw one centred line, colouring only its final impact-word occurrence."""
    if not impact_word or not highlight_this_line:
        _draw_mixed(img, draw, (x, y), line, font, emoji_font,
                    fill=text_fill, outline=outline)
        return

    matches = list(re.finditer(re.escape(impact_word), line, re.IGNORECASE))
    if not matches:
        _draw_mixed(img, draw, (x, y), line, font, emoji_font,
                    fill=text_fill, outline=outline)
        return
    match = matches[-1]
    pieces = (
        (line[:match.start()], text_fill),
        (line[match.start():match.end()], highlight_fill),
        (line[match.end():], text_fill),
    )
    cursor = float(x)
    for piece, fill in pieces:
        if not piece:
            continue
        _draw_mixed(img, draw, (cursor, y), piece, font, emoji_font,
                    fill=fill, outline=outline)
        cursor += _measure_width(draw, piece, font, emoji_font)


def _draw_accent_hook(text, target_width, font_path, font_size,
                      output_image_path, look):
    """Render the default two-line white/gold hook on a subtle dark panel."""
    # Emoji use the platform font when available. If the container has none,
    # remove them instead of drawing empty tofu squares, matching other styles.
    emoji_font = _load_emoji_font(font_size) if _EMOJI_RE.search(text) else None
    if _EMOJI_RE.search(text) and emoji_font is None:
        text = re.sub(r"[ \t]{2,}", " ", _EMOJI_RE.sub("", text)).strip()

    max_text_width = max(1, int(target_width - font_size * 0.72))
    min_font_size = max(24, int(font_size * 0.70))
    while True:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()
        emoji_font = _load_emoji_font(font_size) if _EMOJI_RE.search(text) else None
        probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        lines = _balanced_accent_lines(text, probe, font, emoji_font, max_text_width)
        widest = max(_measure_width(probe, line, font, emoji_font) for line in lines)
        if widest <= max_text_width or font_size <= min_font_size:
            break
        font_size = max(min_font_size, font_size - 2)

    pad_x = int(font_size * 0.38)
    pad_y = int(font_size * 0.28)
    line_gap = int(font_size * 0.12)
    margin = max(18, int(font_size * 0.30))
    widths = [int(_measure_width(probe, line, font, emoji_font)) for line in lines]
    boxes = [probe.textbbox((0, 0), line or "Ágj", font=font) for line in lines]
    heights = [max(box[3] - box[1], font_size) for box in boxes]
    # Render the complete text even when one pathological word cannot fit at
    # the minimum font size. The finished overlay is scaled below; clipping it
    # to target_width here would permanently cut characters off.
    panel_w = max(widths) + 2 * pad_x
    panel_h = sum(heights) + line_gap * (len(lines) - 1) + 2 * pad_y
    canvas_w = int(panel_w + 2 * margin)
    canvas_h = int(panel_h + 2 * margin)

    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    panel_box = (margin, margin, margin + panel_w, margin + panel_h)
    radius = int(font_size * 0.28)

    # Soft shadow, kept separate from the translucent panel so dark footage
    # remains readable without creating an opaque caption box.
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle(
        (panel_box[0] + 4, panel_box[1] + 6,
         panel_box[2] + 4, panel_box[3] + 6),
        radius=radius,
        fill=(0, 0, 0, 115),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(max(5, int(font_size * 0.12))))
    img.alpha_composite(shadow)

    # A gentle top-to-bottom alpha gradient rather than a flat black card.
    panel = Image.new("RGBA", (int(panel_w), int(panel_h)), (0, 0, 0, 0))
    panel_draw = ImageDraw.Draw(panel)
    base = look["box"]
    for row in range(max(1, int(panel_h))):
        progress = row / max(panel_h - 1, 1)
        alpha = int(base[3] * (1.0 - 0.22 * progress))
        panel_draw.line((0, row, panel_w, row), fill=(*base[:3], alpha))
    mask = Image.new("L", panel.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, panel_w - 1, panel_h - 1), radius=radius, fill=255)
    panel.putalpha(Image.composite(panel.getchannel("A"), Image.new("L", panel.size, 0), mask))
    img.alpha_composite(panel, (margin, margin))

    impact_span = _impact_word_span(text)
    impact_word = text[impact_span[0]:impact_span[1]] if impact_span else None
    highlight_line = None
    if impact_word:
        for index in range(len(lines) - 1, -1, -1):
            if re.search(re.escape(impact_word), lines[index], re.IGNORECASE):
                highlight_line = index
                break

    draw = ImageDraw.Draw(img)
    current_y = margin + pad_y
    outline = look["outline"]
    for index, (line, line_w, box, line_h) in enumerate(
            zip(lines, widths, boxes, heights)):
        x = (canvas_w - line_w) / 2
        y = current_y - box[1]
        _draw_accent_line(
            img, draw, x, y, line, font, emoji_font, impact_word,
            index == highlight_line, look["text"], look["highlight"], outline)
        current_y += line_h + line_gap

    # ``target_width`` is the complete 9:16 safe-area budget, including the
    # transparent shadow margin—not merely the charcoal panel.
    max_canvas_w = int(target_width)
    if canvas_w > max_canvas_w:
        ratio = max_canvas_w / float(canvas_w)
        canvas_w = max_canvas_w
        canvas_h = max(1, int(round(canvas_h * ratio)))
        img = img.resize((canvas_w, canvas_h), Image.LANCZOS)

    img.save(output_image_path)
    return output_image_path, canvas_w, canvas_h


def _draw_pills(lines, font, emoji_font, font_size, box_fill, text_fill, output_image_path):
    """One rounded box per line, stacked and centred (the "pill" style).

    Every box has the same height (from the font's own ascent/descent plus
    accents), so the stack reads as one block whatever the letters are.
    """
    lines = [ln for ln in lines if ln.strip()] or [" "]
    pad_x = int(font_size * 0.48)
    pad_y = int(font_size * 0.24)
    radius = int(font_size * 0.36)
    probe = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
    top, bottom = probe.textbbox((0, 0), "ÁÉÍÓÚÑgjpqy", font=font)[1::2]
    line_h = (bottom - top) + 2 * pad_y
    widths = [int(_measure_width(probe, ln, font, emoji_font)) for ln in lines]
    canvas_w = max(widths) + 2 * pad_x + 40
    canvas_h = line_h * len(lines) + 40

    img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for i, (ln, w) in enumerate(zip(lines, widths)):
        box_w = w + 2 * pad_x
        x0 = (canvas_w - box_w) // 2
        y0 = 20 + i * line_h
        draw.rounded_rectangle([x0, y0, x0 + box_w, y0 + line_h], radius=radius, fill=box_fill)
        _draw_mixed(img, draw, (x0 + pad_x, y0 + pad_y - top), ln, font, emoji_font,
                    fill=text_fill, outline=None)
    img.save(output_image_path)
    return output_image_path, canvas_w, canvas_h


def create_hook_image(text, target_width, output_image_path="hook_overlay.png", font_scale=1.0, style="pill",
                      font=None):
    """
    Generates a hook overlay image using pixel-based wrapping.
    target_width: The max width the box should occupy (e.g. 85% of video)
    style: one of HOOK_STYLES. ``pill`` is the backwards-compatible key for
      the modern default white/gold accent treatment.
    """
    download_font_if_needed()

    look = HOOK_STYLES.get(style, HOOK_STYLES["pill"])
    box_fill = look["box"]
    text_fill = look["text"]
    outline = look["outline"]
    has_box = box_fill[3] > 0
    draw_shadow = look["shadow"]
    
    # Configuration
    padding_x = 30 # Balanced padding
    padding_y = 25 
    line_spacing = 20 # Increased spacing
    cornerradius = 20
    shadow_offset = (5, 5) 
    shadow_blur = 10
    
    # Font Size Calculation (approx 5% of width - tuned to match Noto Serif Bold metrics in browser)
    pills = bool(look.get("pills"))
    # A chosen typeface (HOOK_FONTS key) wins; otherwise the style's own
    # (pill -> Montserrat, the rest -> Noto Serif, as they always rendered).
    default_font = "montserrat" if look.get("font") == PILL_FONT_PATH else "serif"
    font_path, size_factor = HOOK_FONTS.get(font or default_font, HOOK_FONTS[default_font])
    base_font_size = int(target_width * size_factor)
    font_size = int(base_font_size * font_scale)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"⚠️ Warning: Could not load font {font_path}, using default. Error: {e}")
        font = ImageFont.load_default()

    if look.get("accent"):
        # add_hook_to_video gives styles a 90%-of-frame budget. The modern
        # default uses only 74% of the complete frame, leaving generous side
        # space so it reads as a centred title rather than a subtitle band.
        accent_width = max(1, int(target_width * (74 / 90)))
        return _draw_accent_hook(
            text, accent_width, font_path, font_size, output_image_path, look)

    # Emoji handling: render with an emoji-capable font if one exists,
    # otherwise strip emoji instead of drawing tofu boxes.
    emoji_font = None
    if _EMOJI_RE.search(text):
        emoji_font = _load_emoji_font(font_size)
        if emoji_font is None:
            text = _EMOJI_RE.sub("", text)
            text = re.sub(r"[ \t]{2,}", " ", text).strip()

    # Wrap text logic (Pixel-based)
    dummy_img = Image.new('RGBA', (1, 1))
    draw = ImageDraw.Draw(dummy_img)

    max_text_width = target_width - (2 * padding_x)

    # Handle manual newlines first
    paragraphs = text.split('\n')
    lines = []

    for p in paragraphs:
        if not p.strip():
            lines.append("")
            continue

        words = p.split()
        current_line = []

        for word in words:
            # Test if adding word fits
            test_line = ' '.join(current_line + [word])
            w = _measure_width(draw, test_line, font, emoji_font)

            if w <= max_text_width:
                current_line.append(word)
                continue

            # Word doesn't fit on the current line
            if current_line:
                lines.append(' '.join(current_line))
                current_line = []

            if _measure_width(draw, word, font, emoji_font) <= max_text_width:
                current_line = [word]
            else:
                # Single word wider than the box: hard-wrap it character-wise
                # so it can't get cut off at the edges.
                pieces = _break_long_word(draw, word, font, emoji_font, max_text_width)
                lines.extend(pieces[:-1])
                current_line = [pieces[-1]] if pieces else []

        if current_line:
            lines.append(' '.join(current_line))

    if pills:
        return _draw_pills(lines, font, emoji_font, font_size, box_fill, text_fill,
                           output_image_path)

    # Recalculate true width/height
    max_line_width = 0
    text_heights = []

    for line in lines:
        if not line:
            text_heights.append(font_size) # Use font size for empty line height
            continue

        w = _measure_width(draw, line, font, emoji_font)
        bbox = draw.textbbox((0, 0), line, font=font)
        h = bbox[3] - bbox[1]
        max_line_width = max(max_line_width, int(w))
        text_heights.append(h)
    
    # Box dimensions
    # We want the box to fit the text exactly + padding
    # Ensure min width for aesthetic reasons if text is short (at least 30% of target)
    box_width = max(max_line_width + (2 * padding_x), int(target_width * 0.3))
    
    # Total Text Height: sum(heights) + spacing * (n-1)
    if not text_heights:
         total_text_height = font_size
    else:
         total_text_height = sum(text_heights) + (len(text_heights) - 1) * line_spacing
         
    box_height = total_text_height + (2 * padding_y)
    
    # Create Final Image with Rounded Corners and Shadow
    # 1. Canvas for Shadow (larger than box)
    canvas_w = box_width + 40
    canvas_h = box_height + 40
    
    img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 2. Draw Shadow (only for boxed styles)
    if draw_shadow and has_box:
        shadow_box = [
            (20 + shadow_offset[0], 20 + shadow_offset[1]),
            (20 + box_width + shadow_offset[0], 20 + box_height + shadow_offset[1])
        ]
        draw.rounded_rectangle(shadow_box, radius=cornerradius, fill=(0, 0, 0, 100))
        # 3. Blur Shadow
        img = img.filter(ImageFilter.GaussianBlur(5))

    # 4. Draw Box (sharper, on top of blurred shadow)
    draw_final = ImageDraw.Draw(img)

    if has_box:
        main_box = [
            (20, 20),
            (20 + box_width, 20 + box_height)
        ]
        draw_final.rounded_rectangle(main_box, radius=cornerradius, fill=box_fill)

    # 5. Draw Text
    current_y = 20 + padding_y - 2 # Minor visual adjustment
    for i, line in enumerate(lines):
        if not line:
            current_y += font_size + line_spacing
            continue

        line_w = _measure_width(draw_final, line, font, emoji_font)
        bbox = draw_final.textbbox((0, 0), line, font=font)
        line_h = text_heights[i] if i < len(text_heights) else bbox[3] - bbox[1]

        # Center X
        x = 20 + int(box_width - line_w) // 2

        # Draw text in the style's color (emoji runs use the emoji font)
        _draw_mixed(img, draw_final, (x, current_y), line, font, emoji_font,
                    fill=text_fill, outline=outline)

        current_y += line_h + line_spacing
        
    img.save(output_image_path)
    return output_image_path, canvas_w, canvas_h

def _auto_place(video_path, text, target_box_width, hook_filename, font_scale, style, font,
                video_width, video_height, duration, ranges, has_captions, first_dims):
    """position='auto': pick the height (and, only if every height covers a
    face, a smaller text) with hook_placement. Returns {"image": (path, w, h),
    "y_px", "y", "scale"}, or None to fall back to the legacy top spot (no
    readable frames, no MediaPipe, or nothing fits between the bands)."""
    import hook_placement
    import layout_ranges as _layouts

    window_end = float(duration) if duration else hook_placement.DEFAULT_WINDOW
    if ranges is None:
        ranges = _layouts.read(video_path)
    bands = hook_placement.caption_bands(_layouts.normalise(ranges), 0.0, window_end, has_captions)
    frames = hook_placement.detect_faces(video_path, 0.0, window_end)
    if frames is None:
        print("   ℹ️ Hook placement: no frames read, keeping the top spot.")
        return None

    dims = {1.0: first_dims}

    def measure(rel):
        if rel not in dims:
            _, w, h = create_hook_image(text, target_box_width, hook_filename,
                                        font_scale=font_scale * rel, style=style, font=font)
            dims[rel] = (w, h)
        w, h = dims[rel]
        return w / video_width, h / video_height

    rel, y, cost = hook_placement.best_scale(measure, frames, bands)
    if y is None:
        print("   ℹ️ Hook placement: no spot clears the captions, keeping the top spot.")
        return None
    # The temp PNG holds the last size measured: render the chosen one again.
    _, w, h = create_hook_image(text, target_box_width, hook_filename,
                                font_scale=font_scale * rel, style=style, font=font)
    n_faces = sum(len(f) for f in frames)
    print(f"   🎯 Hook placement: y={y:.2f} scale={rel:g} face cover={cost:.2f} "
          f"({n_faces} faces in {len(frames)} frames, bands={bands})")
    return {"image": (hook_filename, w, h), "y_px": int(round(y * video_height)),
            "y": round(y, 3), "scale": rel}


def add_hook_to_video(video_path, text, output_path, position="top", font_scale=1.0, duration=None, style="pill",
                      also=None, font=None, layout_ranges=None, has_captions=True,
                      mark=None):
    """
    Overlays text hook onto video.
    position: 'top', 'center', 'bottom', or 'auto' (hook_placement: off the
      faces in the hook's window and off the caption bands; what the job
      pipeline and an editor left on "auto" use). An explicit top/center/
      bottom is drawn exactly where it always was.
    layout_ranges / has_captions: only read by 'auto', to know where the
      captions sit (seam on SPLIT stretches, bottom elsewhere). None reads
      the clip's own ``.layout.json`` sidecar.
    font_scale: float multiplier (1.0 = default)
    style: hook look (see HOOK_STYLES)
    also: optional (vf, path): ALSO write ``path`` = the hooked picture with
      ``vf`` applied (the captions), from the same decode. The job pipeline
      needs both files (the editor re-captions from the hooked one), and one
      ffmpeg with two outputs saves a full decode + pass per clip.
    mark: optional (logo_path, make_filter, path), only with ``also``: ALSO
      write ``path`` = the captioned picture with the free-plan mark, from the
      same decode (the served ``wm_`` copy, see main.mark_delivery).
      ``make_filter(vw, vh, video=, logo=, out=)`` is main.watermark_filter.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video {video_path} not found")

    # 1. Probe video width to scale text properly
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', video_path]
        res = subprocess.check_output(cmd, timeout=60).decode().strip()
        # Takes first stream if multiple
        dims = res.split('\n')[0].split('x')
        video_width = int(dims[0])
        video_height = int(dims[1])
    except Exception as e:
        print(f"⚠️ FFprobe failed: {e}. Assuming 1080x1920")
        video_width = 1080
        video_height = 1920
        
    # 2. Generate Image
    # Box check: Don't let it be wider than 90% of screen
    target_box_width = int(video_width * 0.9)
    
    # Unique per invocation so parallel jobs can't overwrite each other's overlay.
    # The uuid alone guarantees that; the source name rides along only as a
    # debugging hint, so it is trimmed by BYTES (filesystems cap at 255 bytes,
    # and one Bengali or Arabic character costs three). Embedding it untrimmed
    # raised OSError 36 and killed the endpoint in prod on 26-jul-2026.
    stem = os.path.splitext(os.path.basename(video_path))[0]
    hook_filename = (f"temp_hook_{uuid.uuid4().hex[:8]}_"
                     f"{_truncate_bytes(stem, 80)}.png")
    
    try:
        img_path, box_w, box_h = create_hook_image(text, target_box_width, hook_filename, font_scale=font_scale, style=style,
                                                       font=font)
        placed = None
        if position == "auto":
            placed = _auto_place(video_path, text, target_box_width, hook_filename, font_scale,
                                 style, font, video_width, video_height, duration,
                                 layout_ranges, has_captions, (box_w, box_h))
            if placed:
                img_path, box_w, box_h = placed["image"]

        # 3. Calculate Overlay Position
        overlay_x = (video_width - box_w) // 2
        
        if placed:
            overlay_y = placed["y_px"]
        elif position == "center":
            overlay_y = (video_height - box_h) // 2
        elif position == "bottom":
             # Bottom 20% mark (approx)
             overlay_y = int(video_height * 0.70)
        else:
             # Top 20% mark
             overlay_y = int(video_height * 0.20)
        
        # 4. FFmpeg Command
        print(f"🎬 Overlaying hook: '{text}' at {overlay_x},{overlay_y}")
        
        overlay = (f"[0:v][1:v]overlay={overlay_x}:{overlay_y}"
                   + (f":enable='between(t,0,{float(duration)})'" if duration else ""))
        outputs = [output_path]
        if also:
            outputs.append(also[1])
            if mark:
                outputs.append(mark[2])

        def build(encode_args):
            tail = ['-c:a', 'copy', *encode_args, *METADATA_SCRUB,
                    '-movflags', '+faststart']
            if also and mark:
                extra_vf, extra_path = also
                logo_path, make_filter, mark_path = mark
                marked = make_filter(
                    video_width, video_height, video="[sm]",
                    logo="[2:v]", out="[m]")
                return [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-i', img_path,
                    '-i', logo_path,
                    '-filter_complex',
                    f"{overlay},split=2[h][c];"
                    f"[c]{extra_vf},split=2[s][sm];{marked}",
                    '-map', '[h]', '-map', '0:a?', *tail, output_path,
                    '-map', '[s]', '-map', '0:a?', *tail, extra_path,
                    '-map', '[m]', '-map', '0:a?', *tail, mark_path,
                ]
            if also:
                extra_vf, extra_path = also
                return [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-i', img_path,
                    '-filter_complex', f"{overlay},split=2[h][c];[c]{extra_vf}[s]",
                    '-map', '[h]', '-map', '0:a?', *tail, output_path,
                    '-map', '[s]', '-map', '0:a?', *tail, extra_path,
                ]
            return [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', img_path,
                '-filter_complex', overlay,
                *tail,
                output_path,
            ]

        def complete(result):
            files_ok = all(
                os.path.isfile(path) and os.path.getsize(path) > 0
                for path in outputs
            )
            return (result.returncode == 0 and files_ok) or finished_before_crash(
                result, outputs)

        def execute(command):
            return gpu_decode.run(
                lambda on_gpu: (
                    gpu_decode.on_card(command, video_path) if on_gpu else command
                ),
                lambda candidate: subprocess.run(
                    candidate, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    timeout=1800),
                ok=complete,
                path=video_path,
            )

        def remove_partial_outputs():
            for path in outputs:
                if os.path.isfile(path):
                    os.remove(path)

        result = run_encode(
            build,
            execute,
            complete,
            remove_partial_outputs,
            tier=QUALITY,
        )
        if not complete(result):
            raise subprocess.CalledProcessError(result.returncode, result.args,
                                                result.stdout, result.stderr)
        print(f"✅ Hook added to {output_path}")
        if placed:
            return {"position": "auto", "y": placed["y"], "scale": placed["scale"],
                    "height": round(box_h / video_height, 3)}
        return True

    except subprocess.TimeoutExpired:
        print("❌ FFmpeg hook overlay timed out after 1800s.")
        raise RuntimeError("FFmpeg hook overlay timed out after 1800s.")
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg Error: {e.stderr.decode() if e.stderr else 'Unknown'}")
        raise e
    except Exception as e:
        print(f"❌ Hook Gen Error: {e}")
        raise e
    finally:
        # Cleanup temp image
        if os.path.exists(hook_filename):
            os.remove(hook_filename)
