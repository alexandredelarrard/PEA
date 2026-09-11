"""
cube_evidence_pdf.py  (scripts/)
--------------------------------------------------------------------------------------------
PDF renderer for the `cube_part_fundamentals` evidence report.

Landscape A3, because the per-feature table is 8 columns wide and three of them carry prose:
on A4 the "why it is right" column collapses to about four characters per line and the table
becomes unreadable. A3 gives ~1120pt of usable width, which is enough for the three prose
columns to run 40-60 characters.

⚠ FONTS. The report is full of characters Latin-1 does not have -- `⚠ ≥ ≈ × σ — →` -- and
reportlab's built-in Helvetica silently renders every one of them as a black box. matplotlib
is already a declared dependency and ships DejaVu, which covers all of them, so its font files
are registered here rather than adding a font dependency of our own. If that lookup ever
fails the fallback is Helvetica AND the offending characters are transliterated, so the
document degrades to plain ASCII instead of to boxes.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

PAGE = landscape(A3)
MARGIN = 12 * mm
USABLE = PAGE[0] - 2 * MARGIN

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5b5b5b")
RULE = colors.HexColor("#c8c8c8")
BAND = colors.HexColor("#f2f4f7")
HEAD = colors.HexColor("#e3e8ef")
GOOD = colors.HexColor("#1a7f37")
WARN = colors.HexColor("#b35309")

#: characters that must survive; used only to test whether the registered font has them
_PROBE = "⚠≥≈×σ—→⊂✅"
#: last-resort transliteration when no unicode-capable font is available
_ASCII = {"⚠": "!", "≥": ">=", "≤": "<=", "≈": "~", "×": "x", "σ": "sd", "—": "--",
          "–": "-", "→": "->", "⊂": "subset of", "…": "...", "✅": "[x]", "⬜": "[ ]",
          "🔄": "[~]", "’": "'", "‘": "'", "“": '"', "”": '"', "ℹ": "i", "·": "-"}

BASE, BOLD, MONO = "Helvetica", "Helvetica-Bold", "Courier"
_UNICODE_OK = False


def _register_fonts() -> None:
    """Register DejaVu from matplotlib's bundled font directory.

    ⚠ `registerFontFamily` IS NOT OPTIONAL. Registering the four faces individually is
    enough to SET a font, but `<b>` and `<i>` inside a Paragraph are resolved through the
    family map — without it reportlab finds no bold face for `DejaVuSans` and silently
    renders the markup as plain text. Every `**bold**` in this report came out flat until
    the family was declared."""
    global BASE, BOLD, MONO, _UNICODE_OK
    try:
        import matplotlib
        d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        faces = {"DejaVuSans": "DejaVuSans.ttf",
                 "DejaVuSans-Bold": "DejaVuSans-Bold.ttf",
                 "DejaVuSans-Oblique": "DejaVuSans-Oblique.ttf",
                 "DejaVuSans-BoldOblique": "DejaVuSans-BoldOblique.ttf",
                 "DejaVuSansMono": "DejaVuSansMono.ttf",
                 "DejaVuSansMono-Bold": "DejaVuSansMono-Bold.ttf"}
        for name, fn in faces.items():
            path = d / fn
            if not path.exists():
                return
            pdfmetrics.registerFont(TTFont(name, str(path)))
        pdfmetrics.registerFontFamily(
            "DejaVuSans", normal="DejaVuSans", bold="DejaVuSans-Bold",
            italic="DejaVuSans-Oblique", boldItalic="DejaVuSans-BoldOblique")
        pdfmetrics.registerFontFamily(
            "DejaVuSansMono", normal="DejaVuSansMono", bold="DejaVuSansMono-Bold",
            italic="DejaVuSansMono", boldItalic="DejaVuSansMono-Bold")
        BASE, BOLD, MONO = "DejaVuSans", "DejaVuSans-Bold", "DejaVuSansMono"
        _UNICODE_OK = True
    except Exception:                                  # pragma: no cover - font env
        pass


def _text(s: str, mono_size: float | None = None) -> str:
    """Catalogue prose -> reportlab's mini-HTML.

    `code` -> monospace, **bold** -> bold, *emphasis* -> italic. Escaping happens FIRST so a
    literal `<` in a formula cannot open a tag; the markup substitutions then insert the only
    real tags.

    `mono_size` scales the inline code face to the surrounding text. It is a parameter and
    not a constant because DejaVu Sans Mono runs visibly larger than the proportional face at
    the same point size, so a fixed 6pt that reads correctly in a 6.4pt table cell looks like
    a typo inside a 20pt title."""
    s = str(s)
    if not _UNICODE_OK:
        for k, v in _ASCII.items():
            s = s.replace(k, v)
    s = s.replace("\\|", "|")                          # undo the markdown table escaping
    s = html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    # single-asterisk emphasis, but only where it is not a leftover of the bold pass and not
    # a bare arithmetic `*` (which the catalogue does use, e.g. `grossMargins * revenue`)
    s = re.sub(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", s)
    size = f' size="{mono_size:.1f}"' if mono_size else ""
    s = re.sub(r"`([^`]+)`", rf'<font face="{MONO}"{size}>\1</font>', s)
    return s


def _styles() -> dict:
    ss = getSampleStyleSheet()
    mk = lambda n, **kw: ParagraphStyle(n, parent=ss["BodyText"], **kw)  # noqa: E731
    return {
        "title": mk("t", fontName=BOLD, fontSize=20, leading=24, textColor=INK,
                    spaceAfter=2),
        "subtitle": mk("st", fontName=BASE, fontSize=10, leading=14, textColor=MUTED,
                       spaceAfter=10),
        "h1": mk("h1", fontName=BOLD, fontSize=14, leading=18, textColor=INK,
                 spaceBefore=14, spaceAfter=6),
        "h2": mk("h2", fontName=BOLD, fontSize=10.5, leading=14, textColor=INK,
                 spaceBefore=10, spaceAfter=4),
        "fam": mk("fam", fontName=BOLD, fontSize=9.5, leading=12, textColor=colors.white,
                  backColor=colors.HexColor("#33415c"), borderPadding=(3, 4, 3, 4),
                  spaceBefore=9, spaceAfter=3),
        "body": mk("b", fontName=BASE, fontSize=8.6, leading=12, textColor=INK,
                   alignment=TA_LEFT, spaceAfter=5),
        "cell": mk("c", fontName=BASE, fontSize=6.4, leading=8, textColor=INK,
                   spaceAfter=0),
        # right alignment must live on the PARAGRAPH: a Table `ALIGN` command positions the
        # cell's flowable in the cell, and a Paragraph always fills the full cell width, so
        # the text inside it never moves. Every numeric column here was left-ragged until
        # these styles existed.
        "cellr": mk("cr", fontName=BASE, fontSize=6.4, leading=8, textColor=INK,
                    alignment=TA_RIGHT, spaceAfter=0),
        "cellm": mk("cm", fontName=MONO, fontSize=6.2, leading=8, textColor=INK),
        "cellmr": mk("cmr", fontName=MONO, fontSize=6.2, leading=8, textColor=INK,
                     alignment=TA_RIGHT),
        "th": mk("th", fontName=BOLD, fontSize=6.8, leading=8.5, textColor=INK),
        "thr": mk("thr", fontName=BOLD, fontSize=6.8, leading=8.5, textColor=INK,
                  alignment=TA_RIGHT),
        "famcell": mk("fc", fontName=BOLD, fontSize=9.5, leading=12,
                      textColor=colors.white),
    }


def _grid(extra: list | None = None) -> TableStyle:
    cmds = [
        ("GRID", (0, 0), (-1, -1), 0.25, RULE),
        ("BACKGROUND", (0, 0), (-1, 0), HEAD),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND]),
    ]
    return TableStyle(cmds + (extra or []))


def _para(s, st) -> Paragraph:
    return Paragraph(_text(s, mono_size=max(5.5, st.fontSize - 0.6)), st)


def _simple_table(header: list[str], rows: list[list], widths: list[float],
                  S: dict, aligns: dict[int, str] | None = None) -> Table:
    right = {i for i, a in (aligns or {}).items() if a == "RIGHT"}
    body = [[_para(h, S["thr" if i in right else "th"]) for i, h in enumerate(header)]]
    for r in rows:
        body.append([c if isinstance(c, Paragraph)
                     else _para(c, S["cellr" if i in right else "cell"])
                     for i, c in enumerate(r)])
    t = Table(body, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(_grid())
    return t


# --------------------------------------------------------------------------- #
# the feature table                                                            #
# --------------------------------------------------------------------------- #
_FEAT_COLS = ["feature", "views", "null", "peer-z p1 / p50 / p99", "@clip",
              "what it does", "why it is right", "the tail"]
_FEAT_W = [96, 62, 34, 80, 32, 244, 300, 274]


def _feature_tables(rows, S: dict) -> list:
    """One table per family.

    THE FAMILY NAME IS ROW 0 OF THE TABLE, not a heading above it, and `repeatRows=2` carries
    both it and the column header onto every continuation page. As a separate flowable the
    band orphaned at the foot of a page with its table overleaf, and a family that spilled
    across a break lost its label entirely — the reader saw an unattributed block of rows."""
    out: list = []
    scale = USABLE / sum(_FEAT_W)
    widths = [w * scale for w in _FEAT_W]
    ncol = len(_FEAT_COLS)
    for fam, items in rows:
        band = [Paragraph(_text(fam), S["famcell"])] + [""] * (ncol - 1)
        body = [band, [_para(h, S["thr" if h in ("null", "@clip") else "th"])
                       for h in _FEAT_COLS]]
        for x in items:
            body.append([
                _para(x["characteristic"], S["cellm"]),
                _para(x["views"], S["cell"]),
                _para(x["null"], S["cellr"]),
                _para(x["dist"], S["cellm"]),
                _para(x["clip"], S["cellr"]),
                _para(x["what"], S["cell"]),
                _para(x["why"], S["cell"]),
                _para(x["tail"], S["cell"]),
            ])
        t = Table(body, colWidths=widths, repeatRows=2, hAlign="LEFT")
        t.setStyle(_grid([
            ("SPAN", (0, 0), (-1, 0)),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#33415c")),
            ("BACKGROUND", (0, 1), (-1, 1), HEAD),
            ("TOPPADDING", (0, 0), (-1, 0), 4),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
            # the zebra must start below BOTH header rows or it repaints them
            ("ROWBACKGROUNDS", (0, 2), (-1, -1), [colors.white, BAND]),
        ]))
        out.append(t)
        out.append(Spacer(1, 7))
    return out


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(BASE, 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 8 * mm,
                      "cube_part_fundamentals — per-feature evidence · 2026-09-05")
    canvas.drawRightString(PAGE[0] - MARGIN, 8 * mm, f"page {canvas.getPageNumber()}")
    canvas.setStrokeColor(RULE)
    canvas.line(MARGIN, 11 * mm, PAGE[0] - MARGIN, 11 * mm)
    canvas.restoreState()


def render(out_path: Path, meta: dict, feature_rows: list, narrative) -> Path:
    """Build the PDF. `narrative` is the module holding the hand-written sections."""
    _register_fonts()
    S = _styles()
    doc = SimpleDocTemplate(
        str(out_path), pagesize=PAGE,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=16 * mm,
        title="cube_part_fundamentals — per-feature evidence",
        author="PEA / stock_pick_strat", subject="Feature acceptance evidence")

    story: list = [
        Paragraph(_text("`cube_part_fundamentals` — per-feature evidence", mono_size=17),
                  S["title"]),
        _para(meta["subtitle"], S["subtitle"]),
    ]
    for block in narrative.opening(meta):
        story += _block(block, S)

    story.append(PageBreak())
    story.append(Paragraph(_text("2. Feature-by-feature evidence"), S["h1"]))
    story.append(_para(meta["table_lede"], S["body"]))
    story += _feature_tables(feature_rows, S)

    story.append(PageBreak())
    for block in narrative.closing(meta):
        story += _block(block, S)

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return out_path


def _block(block: tuple, S: dict) -> list:
    """(kind, payload) -> flowables. Kinds: h1, h2, p, table, spacer, pagebreak."""
    kind = block[0]
    if kind == "h1":
        return [Paragraph(_text(block[1]), S["h1"])]
    if kind == "h2":
        return [Paragraph(_text(block[1]), S["h2"])]
    if kind == "p":
        return [_para(block[1], S["body"])]
    if kind == "spacer":
        return [Spacer(1, block[1])]
    if kind == "pagebreak":
        return [PageBreak()]
    if kind == "table":
        _, header, rows, weights, aligns = block
        total = sum(weights)
        widths = [USABLE * w / total for w in weights]
        t = _simple_table(header, rows, widths, S, aligns)
        # only SHORT tables are held together. A 12-row table of prose is taller than the
        # space usually left on a page, so `KeepTogether` pushed it whole and left half a
        # page blank; it has a repeating header, so splitting it costs the reader nothing.
        return [KeepTogether([t]) if len(rows) <= 8 else t, Spacer(1, 6)]
    raise ValueError(f"unknown narrative block: {kind}")
