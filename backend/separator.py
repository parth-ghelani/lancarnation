"""
Position Screen-Print Separator - core engine.

Public API:
    generate_separation_pdf(input_path, output_path=None, max_colors=4) -> bytes

Takes a raster or vector artwork file and produces a print-ready multi-page PDF:
    Page 1        : Cream/paper background rectangle (same size as design canvas)
    Page 2..N-1   : One vector layer per detected ink color (darkest first)
    Page N        : Composite preview of all layers stacked

Every page shares an identical registration frame (corner brackets + crosshair
plus marks). Zero text, zero numbers on any page. Layers are perfectly aligned
by construction since they never leave the source coordinate grid.

Supported inputs: PNG, JPG/JPEG, PDF, SVG.

Dependencies: numpy, Pillow, scikit-learn, scipy, reportlab, svglib, pdf2image,
              plus the potrace binary on PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from reportlab.graphics import renderPDF
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas as pdf_canvas
from scipy import ndimage
from svglib.svglib import svg2rlg


# ---------------------------------------------------------------------------
# Configuration - tweak these if you want, sensible defaults for screen printing
# ---------------------------------------------------------------------------

# Despeckle: connected components smaller than this many pixels are considered
# grain/noise and dropped before color detection.
_MIN_BLOB_AREA = 10

# Only pixels this far from the paper color (Euclidean RGB distance) contribute
# to detecting the "true" ink colors. Anti-aliased edge pixels are excluded
# from color discovery but still get properly alpha-unmixed later.
_CORE_INK_DISTANCE = 55

# Raw ink threshold - the loosest definition of "not paper". Includes soft edges.
_RAW_INK_DISTANCE = 20

# How many pixels to sample from each of the four corners when auto-detecting
# the paper/background color. Bigger is more robust to corner artifacts.
_CORNER_SAMPLE_SIZE = 100

# Registration marks - all measured as fractions of the shorter page dimension
# so they scale with any page size.
_MARGIN_FRAC = 0.045          # gap from paper edge to reg frame
_BRACKET_ARM_FRAC = 0.040     # length of each corner bracket arm
_CROSSHAIR_ARM_FRAC = 0.016   # length of each crosshair arm
_INNER_PAD_FRAC = 0.040       # gap from reg frame to artwork
_LINE_WEIGHT_FRAC = 0.0022    # stroke width for reg marks


@dataclass
class _Layer:
    """One separated ink color, ready to be drawn to the PDF."""

    name: str            # e.g. "layer_1_rgb_76_64_48"
    color: np.ndarray    # (3,) uint8 - the flat ink color
    svg_path: str        # absolute path to the vectorized SVG on disk


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_separation_pdf(
    input_path: str,
    output_path: Optional[str] = None,
    max_colors: int = 4,
    page_size: str = "auto",
    orientation: str = "auto",
) -> bytes:
    """
    Separate an artwork into printable ink layers and return a print-ready PDF.

    Args:
        input_path : Path to the source design. PNG, JPG, PDF, or SVG.
        output_path: Optional path to also write the PDF to disk. If None, the
                     PDF is only returned in memory.
        max_colors : Upper bound on detected ink colors (algorithm picks the
                     best K in [1, max_colors] via an elbow test). Default 4.
        page_size  : "auto" (default) matches the source canvas aspect ratio.
                     Or a fixed size: "A3", "A4", "A5", "Letter", "Legal",
                     "Tabloid". Case-insensitive.
        orientation: "auto" (default) keeps the source's natural orientation.
                     Or force "portrait" / "landscape". Case-insensitive.

    Returns:
        PDF file contents as bytes. Multi-page:
            Page 1        : cream/paper background rectangle
            Page 2..N-1   : one ink color per page (darkest first)
            Page N        : composite preview
        All pages share identical registration marks and coordinate system.

    Raises:
        FileNotFoundError : input_path doesn't exist
        RuntimeError      : potrace not installed on the system
        ValueError        : input format not supported, or bad page_size /
                            orientation string
    """
    input_path = str(input_path)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)
    if shutil.which("potrace") is None:
        raise RuntimeError(
            "potrace binary not found on PATH. Install it (e.g. "
            "'apt install potrace' on Linux, 'brew install potrace' on macOS)."
        )

    page_size = page_size.lower()
    orientation = orientation.lower()
    if orientation not in ("auto", "portrait", "landscape"):
        raise ValueError(f"orientation must be auto/portrait/landscape, got: {orientation}")
    if page_size not in _NAMED_PAGE_SIZES and page_size != "auto":
        raise ValueError(
            f"page_size must be 'auto' or one of {sorted(_NAMED_PAGE_SIZES)}, got: {page_size}"
        )

    with tempfile.TemporaryDirectory() as workdir:
        # 1) Normalize any supported input to a flat RGB raster we can analyze
        rgb_array = _load_as_rgb_array(input_path, workdir)

        # 2) Detect paper + ink colors and split into clean per-color masks
        paper_color, layer_masks, ink_colors = _separate_colors(
            rgb_array, max_colors=max_colors
        )

        # 3) Vectorize each mask -> SVG with the correct ink color
        layers = _vectorize_masks(layer_masks, ink_colors, workdir)

        # 4) Assemble the print-ready PDF (no background layer - the paper /
        # garment is always the substrate the user adds their own base to)
        pdf_bytes = _build_pdf(
            layers=layers,
            canvas_size=(rgb_array.shape[1], rgb_array.shape[0]),
            page_size=page_size,
            orientation=orientation,
        )

    if output_path:
        Path(output_path).write_bytes(pdf_bytes)
    return pdf_bytes


# Named paper sizes in PDF points (1 in = 72 pt), portrait orientation.
# Landscape is just (h, w) swapped in _resolve_page_dimensions below.
_NAMED_PAGE_SIZES = {
    "a3":      (841.89, 1190.55),   # 297 x 420 mm
    "a4":      (595.28, 841.89),    # 210 x 297 mm
    "a5":      (419.53, 595.28),    # 148 x 210 mm
    "letter":  (612.00, 792.00),    # 8.5 x 11 in
    "legal":   (612.00, 1008.00),   # 8.5 x 14 in
    "tabloid": (792.00, 1224.00),   # 11 x 17 in
}


# ---------------------------------------------------------------------------
# Input loading - normalize PNG/JPG/PDF/SVG to a plain RGB numpy array
# ---------------------------------------------------------------------------

def _load_as_rgb_array(input_path: str, workdir: str) -> np.ndarray:
    """Load any supported input and return an (H, W, 3) uint8 RGB array.

    For vector inputs (PDF, SVG), we rasterize at high DPI so the color
    detection has enough pixels to work with. Vectorization at the end still
    produces clean vector output, so this rasterization is lossless in effect.
    """
    ext = os.path.splitext(input_path)[1].lower()

    if ext in (".png", ".jpg", ".jpeg"):
        img = Image.open(input_path).convert("RGB")
        return np.array(img)

    if ext == ".svg":
        # Rasterize the SVG to a large PNG, then load as RGB
        drawing = svg2rlg(input_path)
        if drawing is None:
            raise ValueError(f"Could not parse SVG: {input_path}")
        # Aim for at least 1500px on the long side
        target = 1500
        scale = target / max(drawing.width, drawing.height)
        drawing.width *= scale
        drawing.height *= scale
        drawing.scale(scale, scale)
        tmp_png = os.path.join(workdir, "svg_raster.png")
        renderPDF.drawToFile(drawing, os.path.join(workdir, "_dummy.pdf"))
        # Fall back to using pdf2image via the PDF we just wrote
        return _rasterize_pdf_page(os.path.join(workdir, "_dummy.pdf"))

    if ext == ".pdf":
        return _rasterize_pdf_page(input_path)

    raise ValueError(f"Unsupported input format: {ext}")


def _rasterize_pdf_page(pdf_path: str, dpi: int = 200) -> np.ndarray:
    """Rasterize the first page of a PDF to an (H, W, 3) RGB array."""
    from pdf2image import convert_from_path  # imported lazily
    pages = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)
    if not pages:
        raise ValueError(f"PDF had no rasterizable pages: {pdf_path}")
    return np.array(pages[0].convert("RGB"))


# ---------------------------------------------------------------------------
# Color separation - the core algorithm
# ---------------------------------------------------------------------------

def _separate_colors(
    arr: np.ndarray, max_colors: int
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Detect paper color, ink colors, and produce a clean binary mask per ink.

    Returns:
        paper_color : (3,) float - the detected paper/background RGB
        layer_masks : list of (H, W) bool arrays, one per ink color, no overlap
        ink_colors  : list of (3,) float - the flat ink color for each mask,
                       sorted darkest-first
    """
    arr = arr.astype(np.float32)
    H, W, _ = arr.shape

    # Paper color: median of the four corners is robust against a random dark
    # blob happening to touch a corner.
    c = min(_CORNER_SAMPLE_SIZE, H // 4, W // 4)
    corners = np.concatenate([
        arr[:c, :c].reshape(-1, 3),
        arr[:c, -c:].reshape(-1, 3),
        arr[-c:, :c].reshape(-1, 3),
        arr[-c:, -c:].reshape(-1, 3),
    ])
    paper = np.median(corners, axis=0)

    dist = np.linalg.norm(arr - paper, axis=2)
    raw_ink = dist > _RAW_INK_DISTANCE

    # Despeckle: kill isolated grain-sized blobs, keep real strokes
    labeled, n_blobs = ndimage.label(raw_ink, structure=np.ones((3, 3)))
    if n_blobs > 0:
        sizes = ndimage.sum(raw_ink, labeled, range(1, n_blobs + 1))
        keep_labels = np.where(sizes >= _MIN_BLOB_AREA)[0] + 1
        clean_ink = np.isin(labeled, keep_labels)
    else:
        clean_ink = raw_ink

    # Core-ink pixels = the strongly-colored center of every stroke, well past
    # the anti-aliased edges. These are what we use to DISCOVER the flat ink
    # colors. Edge/soft pixels are still assigned to layers later.
    core_mask = clean_ink & (dist > _CORE_INK_DISTANCE)

    if core_mask.sum() < 100:
        return paper, [], []

    core_pixels = arr[core_mask]

    # Discover ink colors via 3D-histogram peak detection instead of KMeans.
    # KMeans clusters by pixel COUNT, so tiny-but-distinct colors (like the
    # chakra blue in a flag - <1% of pixels) get absorbed into whatever
    # bigger cluster is closest. Peak detection instead finds every color
    # ISLAND in RGB space regardless of size, then merges nearby peaks and
    # drops noise peaks. This handles designs with lots of colors where some
    # are small but visually important.
    ink_colors = _discover_ink_colors_by_peaks(core_pixels, max_colors=max_colors)

    # Assign every pixel in the whole image to the ink color that BEST EXPLAINS
    # its actual RGB value (lowest reconstruction error under the alpha-blend
    # model: pixel = alpha*ink + (1-alpha)*paper). This correctly handles gray
    # anti-aliased edges between dark ink and light paper - they get assigned
    # to the dark ink, not to some random middle-tone cluster.
    recon_errors = []
    for ink in ink_colors:
        v = ink - paper
        denom = float(np.dot(v, v)) + 1e-6
        proj = np.tensordot(arr - paper, v, axes=([2], [0])) / denom
        alpha = np.clip(proj, 0, 1)
        recon = paper[None, None, :] + alpha[..., None] * v[None, None, :]
        err = np.sum((arr - recon) ** 2, axis=2)
        recon_errors.append(err)
    recon_errors = np.stack(recon_errors, axis=0)  # (K, H, W)
    recon_errors[:, ~clean_ink] = np.inf  # never a valid winner outside ink

    winner = np.argmin(recon_errors, axis=0)
    layer_masks: List[np.ndarray] = []
    for k in range(len(ink_colors)):
        mask = (winner == k) & clean_ink
        layer_masks.append(mask)

    # Post-filter: drop layers whose winning-pixel count is negligibly small
    # (< 0.01% of image area OR under 150 px absolute). Handles the case where
    # the user asked for more colors than actually exist - empty layers just
    # fall out. Keep the threshold LOW so tiny-but-real ink colors (small
    # logos, single small elements like the chakra wheel in a flag) survive.
    min_pixels = max(150, int(0.0001 * H * W))
    filtered_masks, filtered_colors = [], []
    for mask, ink in zip(layer_masks, ink_colors):
        if int(mask.sum()) >= min_pixels:
            filtered_masks.append(mask)
            filtered_colors.append(ink)

    return paper, filtered_masks, filtered_colors


def _mode_color(pixels: np.ndarray) -> np.ndarray:
    """Return the most-frequent RGB color in `pixels`, bucketed into 4-unit bins."""
    bucketed = (pixels.astype(np.int32) // 4) * 4
    uniq, counts = np.unique(bucketed, axis=0, return_counts=True)
    return uniq[int(np.argmax(counts))].astype(np.float32)


def _discover_ink_colors_by_peaks(
    pixels: np.ndarray,
    max_colors: int,
    bin_size: int = 32,
    min_peak_count: int = 40,
    merge_distance: float = 55.0,
) -> List[np.ndarray]:
    """Find distinct ink colors via 3D RGB histogram peak detection.

    The idea: real flat ink colors show up as local maxima in RGB space. A
    tiny-but-distinct color (blue chakra) still forms a sharp peak in its own
    corner of RGB space, even if it's <1% of total ink. KMeans misses this
    because it optimizes total variance; peaks care only about local density.

    Algorithm:
      1. Bucket pixels into a coarse 3D RGB grid (bin_size per axis).
      2. Every bin that has more pixels than all six 6-connected neighbors and
         all diagonals is a "peak" - a locally-densest color.
      3. Refine each peak to its true flat color via mode within a wider box.
      4. Reject peaks that are basically grayscale (R==G==B within tolerance)
         AND not near-black or near-white - these are anti-aliasing halos
         between chromatic inks, not real ink colors.
      5. Merge peaks closer than merge_distance in RGB space.
      6. Rank by count and return top max_colors, darkest-first.
    """
    if len(pixels) < 50:
        return []

    px = pixels.astype(np.int32)
    idx = px // bin_size
    max_bin = 256 // bin_size

    # Sparse 3D histogram
    from collections import defaultdict
    hist: dict = defaultdict(int)
    for i in range(len(idx)):
        hist[(int(idx[i, 0]), int(idx[i, 1]), int(idx[i, 2]))] += 1

    # Find peaks: bin count > count of ALL 26 neighbors (6-face + 12-edge + 8-corner)
    peaks = []
    for (r, g, b), count in hist.items():
        if count < min_peak_count:
            continue
        is_peak = True
        for dr in (-1, 0, 1):
            for dg in (-1, 0, 1):
                for db_ in (-1, 0, 1):
                    if dr == 0 and dg == 0 and db_ == 0:
                        continue
                    n = hist.get((r + dr, g + dg, b + db_), 0)
                    if n >= count:
                        is_peak = False
                        break
                if not is_peak:
                    break
            if not is_peak:
                break
        if is_peak:
            peaks.append((r, g, b, count))

    if not peaks:
        top = max(hist.items(), key=lambda kv: kv[1])
        (r, g, b), _ = top
        return [np.array([r * bin_size + bin_size // 2,
                          g * bin_size + bin_size // 2,
                          b * bin_size + bin_size // 2], dtype=np.float32)]

    # Refine each peak to a flat mode color within a widened box.
    refined = []
    for r, g, b, count in peaks:
        low = np.array([r * bin_size, g * bin_size, b * bin_size]) - bin_size // 2
        high = low + bin_size * 2
        low = np.clip(low, 0, 255)
        high = np.clip(high, 0, 255)
        in_box = np.all((px >= low) & (px < high), axis=1)
        if in_box.sum() < min_peak_count:
            continue
        color = _mode_color(pixels[in_box])
        refined.append((color, int(in_box.sum())))

    # SUPPRESS grayscale-halo peaks. A pixel where max(R,G,B) - min(R,G,B) is
    # small is basically grayscale. In flat-color print art, real ink colors
    # are almost always chromatic OR extreme black/extreme white. Mid-tone
    # grays (RGB ~[80,80,80] .. ~[200,200,200]) are almost always halos
    # between a chromatic ink and a dark or light background.
    def is_real_ink(color: np.ndarray) -> bool:
        r, g, b = float(color[0]), float(color[1]), float(color[2])
        chroma = max(r, g, b) - min(r, g, b)
        luma = (r + g + b) / 3
        if chroma > 25:  # clearly chromatic - keep
            return True
        if luma < 30:    # near-black (dark ink or shirt) - keep
            return True
        if luma > 225:   # near-white - keep
            return True
        return False     # gray mid-tone - suppress as halo

    refined = [(c, ct) for c, ct in refined if is_real_ink(c)]

    if not refined:
        return []

    # Sort by count desc so bigger colors take priority in merges
    refined.sort(key=lambda x: -x[1])

    # Merge nearby peaks. Also, a dark chromatic peak that's "on the way to"
    # a brighter chromatic peak already accepted (same hue direction AND much
    # lower brightness AND close-ish spatially) is almost certainly a halo of
    # the brighter peak - merge it in. Be conservative: only merge if BOTH
    # (a) same hue direction AND (b) really close in absolute RGB.
    def is_halo_of(dark: np.ndarray, bright: np.ndarray) -> bool:
        # dark must be strictly darker
        if float(np.linalg.norm(dark)) >= 0.55 * float(np.linalg.norm(bright)):
            return False
        # same hue direction (very tight cosine threshold - these are near-collinear)
        b_norm = float(np.linalg.norm(bright))
        d_norm = float(np.linalg.norm(dark))
        if b_norm < 1 or d_norm < 1:
            return False
        cos = float(np.dot(bright, dark)) / (b_norm * d_norm)
        if cos < 0.985:
            return False
        # AND absolute RGB distance moderately small (otherwise it's a real
        # distinct darker ink, not a halo)
        return float(np.linalg.norm(bright - dark)) < 120.0

    merged: List[Tuple[np.ndarray, int]] = []
    for color, count in refined:
        collision = False
        for kept_color, _ in merged:
            if np.linalg.norm(color - kept_color) < merge_distance:
                collision = True
                break
            if is_halo_of(color, kept_color) or is_halo_of(kept_color, color):
                collision = True
                break
        if not collision:
            merged.append((color, count))
        if len(merged) >= max_colors:
            break

    result = [c for c, _ in merged]
    result.sort(key=lambda c: float(np.linalg.norm(c)))
    return result


# ---------------------------------------------------------------------------
# Vectorization - clean binary mask -> SVG via potrace
# ---------------------------------------------------------------------------

def _vectorize_masks(
    masks: List[np.ndarray],
    ink_colors: List[np.ndarray],
    workdir: str,
) -> List[_Layer]:
    """Run each binary mask through potrace to produce a colored SVG."""
    layers: List[_Layer] = []
    for k, (mask, ink) in enumerate(zip(masks, ink_colors)):
        name = "layer_{}_rgb_{}_{}_{}".format(
            k + 1, int(ink[0]), int(ink[1]), int(ink[2])
        )
        pbm_path = os.path.join(workdir, f"{name}.pbm")
        svg_path = os.path.join(workdir, f"{name}.svg")

        # potrace convention: black pixels are traced, so invert the mask.
        binary = np.where(mask, 0, 255).astype(np.uint8)
        Image.fromarray(binary, mode="L").save(pbm_path)

        subprocess.run(
            [
                "potrace", "-s", "-o", svg_path, pbm_path,
                "--turdsize", "8",       # extra safety despeckle inside potrace
                "--opttolerance", "0.4", # smooth-vs-accurate curve tradeoff
            ],
            check=True,
        )

        # potrace writes fill="#000000" - swap to the real ink color
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(ink[0]), int(ink[1]), int(ink[2])
        )
        with open(svg_path, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace('fill="#000000"', f'fill="{hex_color}"')
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(content)

        layers.append(_Layer(name=name, color=ink.astype(np.uint8), svg_path=svg_path))
    return layers


# ---------------------------------------------------------------------------
# PDF assembly - registration marks + one page per layer + composite preview
# ---------------------------------------------------------------------------

def _resolve_page_dimensions(
    canvas_size: Tuple[int, int],
    page_size: str,
    orientation: str,
) -> Tuple[float, float]:
    """Return (page_w_pt, page_h_pt) after applying page_size + orientation.

    - "auto" page_size: pick a page whose aspect matches the source canvas,
      with the long side at 10 inches.
    - Named page_size: use the canonical (portrait) dimensions.
    - "auto" orientation: keep whatever aspect that produced.
    - "portrait" / "landscape": force short/long side positions.
    """
    src_w, src_h = canvas_size
    src_is_landscape = src_w > src_h

    if page_size == "auto":
        long_side_in = 10.0
        if src_is_landscape:
            page_w = long_side_in * inch
            page_h = long_side_in * inch * (src_h / src_w)
        else:
            page_h = long_side_in * inch
            page_w = long_side_in * inch * (src_w / src_h)
    else:
        page_w, page_h = _NAMED_PAGE_SIZES[page_size]  # canonical portrait

    if orientation == "auto":
        # For named sizes, match the source's orientation. For auto page_size,
        # the aspect already matches - leave as-is.
        if page_size != "auto" and src_is_landscape and page_h > page_w:
            page_w, page_h = page_h, page_w
    elif orientation == "landscape" and page_h > page_w:
        page_w, page_h = page_h, page_w
    elif orientation == "portrait" and page_w > page_h:
        page_w, page_h = page_h, page_w

    return page_w, page_h


def _build_pdf(
    layers: List[_Layer],
    canvas_size: Tuple[int, int],
    page_size: str = "auto",
    orientation: str = "auto",
) -> bytes:
    """Compose the final print-ready PDF and return its bytes.

    Pages: one per detected ink color (darkest first), then a composite preview
    at the end. NO background layer - the paper / garment substrate is the
    user's responsibility to handle separately (they add their own underbase
    or cream block manually when the garment needs it).
    """
    src_w_px, src_h_px = canvas_size
    page_w, page_h = _resolve_page_dimensions(canvas_size, page_size, orientation)

    short_dim = min(page_w, page_h)
    margin = _MARGIN_FRAC * short_dim
    inner_pad = _INNER_PAD_FRAC * short_dim
    bracket_arm = _BRACKET_ARM_FRAC * short_dim
    crosshair_arm = _CROSSHAIR_ARM_FRAC * short_dim
    line_weight = _LINE_WEIGHT_FRAC * short_dim

    # Registration frame (where the bracket corners sit)
    fx0, fy0 = margin, margin
    fx1, fy1 = page_w - margin, page_h - margin
    # Artwork area (inside the frame, with padding)
    ax0, ay0 = fx0 + inner_pad, fy0 + inner_pad
    ax1, ay1 = fx1 - inner_pad, fy1 - inner_pad
    art_w, art_h = ax1 - ax0, ay1 - ay0

    def draw_reg_marks(c: pdf_canvas.Canvas) -> None:
        c.setStrokeColorRGB(0, 0, 0)
        c.setLineWidth(line_weight)
        for x, y, sx, sy in [
            (fx0, fy0, +1, +1),
            (fx1, fy0, -1, +1),
            (fx0, fy1, +1, -1),
            (fx1, fy1, -1, -1),
        ]:
            c.line(x, y, x + sx * bracket_arm, y)
            c.line(x, y, x, y + sy * bracket_arm)

        mx, my = (fx0 + fx1) / 2, (fy0 + fy1) / 2
        for x, y in [(mx, fy0), (mx, fy1), (fx0, my), (fx1, my)]:
            c.line(x - crosshair_arm, y, x + crosshair_arm, y)
            c.line(x, y - crosshair_arm, x, y + crosshair_arm)
            c.circle(x, y, crosshair_arm * 0.55, stroke=1, fill=0)

    def draw_svg(c: pdf_canvas.Canvas, svg_path: str) -> None:
        d = svg2rlg(svg_path)
        scale = min(art_w / d.width, art_h / d.height)
        draw_w = d.width * scale
        draw_h = d.height * scale
        x = ax0 + (art_w - draw_w) / 2
        y = ay0 + (art_h - draw_h) / 2
        c.saveState()
        c.translate(x, y)
        c.scale(scale, scale)
        renderPDF.draw(d, c, 0, 0)
        c.restoreState()

    buf = BytesIO()
    c = pdf_canvas.Canvas(buf, pagesize=(page_w, page_h))

    # One page per detected ink color, darkest first
    for layer in layers:
        draw_reg_marks(c)
        draw_svg(c, layer.svg_path)
        c.showPage()

    # Final page: composite preview - every ink color stacked in real print
    # order (lightest first so darker inks sit on top). No paper block - the
    # user's paper / garment is the visual background.
    draw_reg_marks(c)
    for layer in reversed(layers):
        draw_svg(c, layer.svg_path)
    c.showPage()

    c.save()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI convenience - so you can smoke-test the module from the shell
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Position screen-print separator - one artwork in, print-ready PDF out."
    )
    parser.add_argument("input", help="Input artwork (PNG / JPG / PDF / SVG)")
    parser.add_argument("output", help="Output PDF path")
    parser.add_argument("--max-colors", type=int, default=4, help="Max ink colors to detect (default: 4)")
    parser.add_argument("--page-size", default="auto",
                        help="auto (default) | A3 | A4 | A5 | Letter | Legal | Tabloid")
    parser.add_argument("--orientation", default="auto",
                        help="auto (default) | portrait | landscape")
    args = parser.parse_args()

    generate_separation_pdf(
        args.input, args.output,
        max_colors=args.max_colors,
        page_size=args.page_size,
        orientation=args.orientation,
    )
    print(f"Wrote: {args.output}")
