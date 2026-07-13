"""
Position Screen-Print Separator - core engine.

Public API:
    generate_separation_pdf(input_path, output_path=None, max_colors=4) -> bytes

Algorithm
---------
1. Rasterize the input (PNG/JPG kept as-is; PDF/SVG rasterized at 300 DPI).
2. Build a despeckled ink mask via distance-from-paper + connected-component
   labelling (small noise blobs removed).
3. SHAPE-BASED color detection:
     a. For each surviving connected shape, compute ONE representative color
        (the mean RGB of all pixels in that shape).  A shape with 100 pixels
        and a shape with 100,000 pixels each cast exactly ONE vote — area no
        longer drowns out small distinct shapes.
     b. Cluster the per-shape colors in CIELAB space (perceptually uniform)
        using average-linkage agglomerative clustering with a ΔE threshold.
        Shapes whose representative colors are within the threshold are merged
        into one ink layer; shapes farther apart stay separate.
     c. The final flat ink color for each cluster is the pixel-count-weighted
        mean RGB of its member shapes — larger shapes still govern "what shade
        of navy/yellow is this ink?" once the grouping decision is made.
4. Assign every clean-ink pixel to one ink via reconstruction-error
   minimisation using the alpha-blend model (pixel = α·ink + (1-α)·paper).
   This handles anti-aliased edges regardless of which shape they belong to.
5. Solve α per assigned pixel.  Build one layer per ink:
       RGBA — flat ink color at solved α, fully transparent background.
   Vectorize each layer's thresholded alpha mask with potrace → clean SVG paths.
6. Assemble print-ready PDF: ink layers (darkest first) + composite preview.

Why shape-based beats pixel-based
----------------------------------
Histogram peak-detection weights by raw pixel count.  A gradient-shaded van
painted in soft navy shading spreads its pixels across dozens of slightly
different RGB values; no single value repeats enough to register as a peak, so
the navy disappears entirely.  In contrast, the entire van is ONE connected
shape; its mean color is solidly navy, and it gets one vote in the cluster step
regardless of how many pixels it occupies.  Small distinct shapes (a tiny blue
chakra in a flag, a highlight dot) also get equal votes and can never be
drowned out by a large background shape.

Rules (unchanged)
-----------------
- Background/paper color is NEVER output as an ink layer.
- Layers use the flat ink color at solved α — original pixels never copied.
- Vector output only (potrace).  No rasterized layers in the PDF.
- No text or numbers on any layer — registration marks only.
- Darkest ink first in page order.

Dependencies: numpy, Pillow, scikit-learn, scipy, reportlab, svglib, pdf2image.
Optional:     potrace binary (brew install potrace / apt install potrace).
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from reportlab.graphics import renderPDF
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as pdf_canvas
from scipy import ndimage
from svglib.svglib import svg2rlg

# Allow large print-master artwork files.
Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MIN_BLOB_AREA       = 10
_CORE_INK_DISTANCE   = 55
_RAW_INK_DISTANCE    = 20
_CORNER_SAMPLE_SIZE  = 100
_PAPER_REJECT_RGB    = 35    # RGB distance: any ink candidate closer than this to paper is dropped
_PAPER_REJECT_LAB    = 12.0  # ΔE (LAB) distance for the same guard, perceptual
_MARGIN_FRAC         = 0.045
_BRACKET_ARM_FRAC    = 0.040
_CROSSHAIR_ARM_FRAC  = 0.016
_INNER_PAD_FRAC      = 0.040
_LINE_WEIGHT_FRAC    = 0.0022
_ALPHA_THRESHOLD     = 0.50   # potrace bitmap threshold

# Named paper sizes in PDF points (1 in = 72 pt), portrait orientation.
_NAMED_PAGE_SIZES: Dict[str, Tuple[float, float]] = {
    "a3":      (841.89, 1190.55),
    "a4":      (595.28,  841.89),
    "a5":      (419.53,  595.28),
    "letter":  (612.00,  792.00),
    "legal":   (612.00, 1008.00),
    "tabloid": (792.00, 1224.00),
}


@dataclass
class _Layer:
    """One separated ink color layer."""
    name:     str            # e.g. "layer_1_rgb_76_64_48"
    color:    np.ndarray     # (3,) uint8 — the flat ink color
    svg_path: Optional[str]         = None  # potrace vector output
    image:    Optional[Image.Image] = None  # RGBA fallback


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_separation_pdf(
    input_path:  str,
    output_path: Optional[str] = None,
    max_colors:  int = 4,
    page_size:   str = "auto",
    orientation: str = "auto",
) -> bytes:
    """
    Separate an artwork file into printable ink layers and return a PDF.

    Pages 1..N    — one ink color per page, flat ink color at solved alpha,
                    transparent background (vectorized with potrace when installed).
    Page  N+1     — composite preview (original artwork, all colors together).
    All pages share identical registration marks.

    Args:
        input_path:  PNG, JPG, PDF, or SVG source artwork.
        output_path: Optional path to also write the PDF to disk.
        max_colors:  Upper bound on detected ink colors (default 4).
        page_size:   "auto" (fits source), or "A3/A4/A5/Letter/Legal/Tabloid".
        orientation: "auto", "portrait", or "landscape".

    Returns:
        PDF file contents as bytes.

    Raises:
        FileNotFoundError: input_path does not exist.
        ValueError:        unsupported format or bad parameters.
    """
    input_path = str(input_path)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)

    page_size   = page_size.lower()
    orientation = orientation.lower()
    if orientation not in ("auto", "portrait", "landscape"):
        raise ValueError(
            f"orientation must be auto/portrait/landscape, got: {orientation!r}"
        )
    if page_size not in _NAMED_PAGE_SIZES and page_size != "auto":
        raise ValueError(
            f"page_size must be 'auto' or one of {sorted(_NAMED_PAGE_SIZES)}, "
            f"got: {page_size!r}"
        )

    with tempfile.TemporaryDirectory() as workdir:
        rgb_array   = _load_as_rgb_array(input_path, workdir)
        paper_color, layer_masks, ink_colors = _separate_colors(
            rgb_array, max_colors=max_colors
        )

        if not ink_colors:
            raise ValueError("No ink colors detected in this design.")

        layers = _create_layers(
            rgb_array, paper_color, layer_masks, ink_colors, workdir
        )

        # ABSOLUTE RULE: the composite / preview MUST always render on a WHITE
        # background regardless of the input's background color. Instead of
        # displaying the raw input (which would carry a yellow/green/blue
        # background through), we rebuild the composite by alpha-compositing
        # each ink layer over a pure-white canvas.
        composite_image = _build_composite_on_white(
            rgb_array, paper_color, layer_masks, ink_colors
        )

        pdf_bytes = _build_pdf(
            layers          = layers,
            composite_image = composite_image,
            canvas_size     = (rgb_array.shape[1], rgb_array.shape[0]),
            page_size       = page_size,
            orientation     = orientation,
        )

    if output_path:
        Path(output_path).write_bytes(pdf_bytes)
    return pdf_bytes


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def _load_as_rgb_array(input_path: str, workdir: str) -> np.ndarray:
    """Load any supported input → (H, W, 3) uint8 RGB array."""
    ext = os.path.splitext(input_path)[1].lower()

    if ext in (".png", ".jpg", ".jpeg"):
        return np.array(Image.open(input_path).convert("RGB"))

    if ext == ".svg":
        drawing = svg2rlg(input_path)
        if drawing is None:
            raise ValueError(f"Could not parse SVG: {input_path}")
        # Scale long side to 2400 PDF points; rasterize at 72 DPI → 2400 px.
        target = 2400
        scale  = target / max(drawing.width, drawing.height)
        drawing.width  *= scale
        drawing.height *= scale
        drawing.scale(scale, scale)
        tmp_pdf = os.path.join(workdir, "_raster.pdf")
        renderPDF.drawToFile(drawing, tmp_pdf)
        return _rasterize_pdf_page(tmp_pdf, dpi=72)

    if ext == ".pdf":
        return _rasterize_pdf_page(input_path, dpi=300)

    raise ValueError(f"Unsupported input format: {ext!r}")


def _rasterize_pdf_page(pdf_path: str, dpi: int = 300) -> np.ndarray:
    """Rasterize the first page of a PDF → (H, W, 3) uint8 RGB array."""
    from pdf2image import convert_from_path
    pages = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)
    if not pages:
        raise ValueError(f"PDF had no rasterizable pages: {pdf_path}")
    return np.array(pages[0].convert("RGB"))


# ---------------------------------------------------------------------------
# Color separation — shape-based clustering + reconstruction-error assignment
# ---------------------------------------------------------------------------

# Max number of shapes fed into agglomerative clustering.
# Shapes are sorted by pixel count; the largest N are clustered and the rest
# are assigned to the nearest cluster centroid (in LAB) afterwards.
_MAX_SHAPES_FOR_CLUSTERING = 500

# ΔE (CIELAB Euclidean) below which two shapes are considered the same ink.
# 30 ΔE ≈ "clearly the same color family even with gradient shading."
# At 54 ΔE, dark-brown and terracotta are safely separated.
_LAB_MERGE_DIST = 30.0


def _separate_colors(
    arr: np.ndarray,
    max_colors: int,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Detect paper color, ink colors, and produce one clean binary mask per ink.

    Color discovery uses the shape-based approach:
      1. Each despeckled connected shape → one representative mean-RGB.
      2. Shape colors are clustered in CIELAB (average-linkage, ΔE threshold).
      3. Final ink color per cluster = pixel-count-weighted mean RGB.
    Pixel assignment uses the alpha-blend reconstruction-error model, which
    correctly handles anti-aliased edges regardless of shape membership.

    Returns:
        paper_color : (3,) float32 — the detected paper/background RGB.
        layer_masks : list of (H, W) bool — one mask per ink, no overlap.
        ink_colors  : list of (3,) float32 — flat ink color per mask,
                      sorted darkest-first.
    """
    arr  = arr.astype(np.float32)
    H, W, _ = arr.shape

    # ── Paper color (corner median) ──────────────────────────────────────────
    c       = min(_CORNER_SAMPLE_SIZE, H // 4, W // 4)
    corners = np.concatenate([
        arr[:c,  :c ].reshape(-1, 3),
        arr[:c,  -c:].reshape(-1, 3),
        arr[-c:, :c ].reshape(-1, 3),
        arr[-c:, -c:].reshape(-1, 3),
    ])
    paper = np.median(corners, axis=0)

    # ── Ink mask + despeckle ─────────────────────────────────────────────────
    dist     = np.linalg.norm(arr - paper, axis=2)
    raw_ink  = dist > _RAW_INK_DISTANCE

    labeled, n_blobs = ndimage.label(raw_ink, structure=np.ones((3, 3)))
    if n_blobs > 0:
        sizes       = ndimage.sum(raw_ink, labeled, range(1, n_blobs + 1))
        keep_labels = np.where(sizes >= _MIN_BLOB_AREA)[0] + 1
        clean_ink   = np.isin(labeled, keep_labels)
    else:
        clean_ink = raw_ink

    if clean_ink.sum() < 100:
        return paper, [], []

    # Re-label after despeckle to get clean, contiguous blob IDs.
    labeled_clean, n_clean = ndimage.label(clean_ink, structure=np.ones((3, 3)))

    # ── Shape-based color discovery ──────────────────────────────────────────
    ink_colors = _discover_ink_colors_by_shapes(
        arr, labeled_clean, n_clean, paper, max_colors
    )
    if not ink_colors:
        return paper, [], []

    # ── Reconstruction-error pixel assignment ────────────────────────────────
    # Each clean-ink pixel is assigned to the ink whose alpha-blend model
    # (pixel = α·ink + (1-α)·paper) best explains it.  This handles edges.
    recon_errors = []
    for ink in ink_colors:
        v     = ink - paper
        denom = float(np.dot(v, v)) + 1e-6
        proj  = np.tensordot(arr - paper, v, axes=([2], [0])) / denom
        alpha = np.clip(proj, 0, 1)
        recon = paper[None, None, :] + alpha[..., None] * v[None, None, :]
        err   = np.sum((arr - recon) ** 2, axis=2)
        recon_errors.append(err)

    recon_errors                = np.stack(recon_errors, axis=0)
    recon_errors[:, ~clean_ink] = np.inf

    winner      = np.argmin(recon_errors, axis=0)
    layer_masks = [(winner == k) & clean_ink for k in range(len(ink_colors))]

    # ── Filter negligible layers + hard paper-color guard ────────────────────
    # ABSOLUTE RULE: the paper/background color must NEVER be emitted as an
    # ink layer. We reject any candidate whose color is within a tight
    # tolerance of paper in BOTH sRGB and CIELAB perceptual space.
    min_pixels = max(150, int(0.0001 * H * W))
    paper_lab  = _rgb_to_lab(paper)
    filtered_masks, filtered_colors = [], []
    for mask, ink in zip(layer_masks, ink_colors):
        if int(mask.sum()) < min_pixels:
            continue
        rgb_dist = float(np.linalg.norm(ink - paper))
        lab_dist = float(np.linalg.norm(_rgb_to_lab(ink) - paper_lab))
        if rgb_dist < _PAPER_REJECT_RGB or lab_dist < _PAPER_REJECT_LAB:
            continue  # too close to paper — this IS the background
        filtered_masks.append(mask)
        filtered_colors.append(ink)

    return paper, filtered_masks, filtered_colors


# ---------------------------------------------------------------------------
# CIELAB conversion
# ---------------------------------------------------------------------------

def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert a single (3,) RGB array [0–255 float] to CIELAB (D65 illuminant).

    Uses the standard sRGB → linear-RGB → XYZ → L*a*b* pipeline.
    The result has L in [0, 100] and a*, b* roughly in [−128, 127].
    """
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0

    def _gamma(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = _gamma(r), _gamma(g), _gamma(b)

    # sRGB D65 matrix
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    # Normalise by D65 white point
    X /= 0.95047
    # Y divided by 1.0 (no-op)
    Z /= 1.08883

    def _f(t: float) -> float:
        return t ** (1.0 / 3.0) if t > 0.008856 else 7.787 * t + 16.0 / 116.0

    fx, fy, fz = _f(X), _f(Y), _f(Z)
    L   = 116.0 * fy - 16.0
    a_  = 500.0 * (fx - fy)
    b_  = 200.0 * (fy - fz)

    return np.array([L, a_, b_], dtype=np.float32)


# ---------------------------------------------------------------------------
# Shape-based ink color discovery
# ---------------------------------------------------------------------------

def _discover_ink_colors_by_shapes(
    arr:           np.ndarray,   # (H, W, 3) float32
    labeled_clean: np.ndarray,   # (H, W) int — connected-component labels
    n_clean:       int,
    paper:         np.ndarray,   # (3,) float32
    max_colors:    int,
    lab_merge_dist: float = _LAB_MERGE_DIST,
) -> List[np.ndarray]:
    """
    Shape-based ink color discovery.

    One mean-RGB data point per connected shape → cluster in LAB → weighted
    mean RGB per cluster.  See module docstring for full rationale.
    """
    from sklearn.cluster import AgglomerativeClustering

    # ── 1. Per-shape representative color ────────────────────────────────────
    shape_rgbs: List[np.ndarray] = []
    shape_pxns: List[int]        = []

    for lbl in range(1, n_clean + 1):
        blob_mask = labeled_clean == lbl
        n_px      = int(blob_mask.sum())
        if n_px < _MIN_BLOB_AREA:
            continue
        mean_rgb = arr[blob_mask].mean(axis=0)
        # Shapes whose mean is essentially paper are background artefacts.
        if float(np.linalg.norm(mean_rgb - paper)) < _RAW_INK_DISTANCE * 0.8:
            continue
        shape_rgbs.append(mean_rgb)
        shape_pxns.append(n_px)

    if not shape_rgbs:
        return []

    # ── 2. Cap shapes fed into clustering; assign the rest afterwards ────────
    order     = sorted(range(len(shape_rgbs)), key=lambda i: shape_pxns[i], reverse=True)
    shape_rgbs = [shape_rgbs[i] for i in order]
    shape_pxns = [shape_pxns[i] for i in order]

    n_shapes   = len(shape_rgbs)
    n_clust    = min(n_shapes, _MAX_SHAPES_FOR_CLUSTERING)
    clust_rgbs = shape_rgbs[:n_clust]
    clust_pxns = shape_pxns[:n_clust]

    # ── 3. Agglomerative clustering in CIELAB ────────────────────────────────
    clust_labs = np.array([_rgb_to_lab(rgb) for rgb in clust_rgbs])  # (N, 3)

    if n_clust == 1:
        cluster_labels = np.array([0])
    else:
        try:
            agg = AgglomerativeClustering(
                distance_threshold = lab_merge_dist,
                n_clusters         = None,
                metric             = "euclidean",
                linkage            = "average",
            )
            cluster_labels = agg.fit_predict(clust_labs)
        except Exception:
            cluster_labels = np.zeros(n_clust, dtype=int)

    # ── 4. Merge excess clusters down to max_colors ──────────────────────────
    if len(set(cluster_labels)) > max_colors:
        cluster_labels = _merge_closest_clusters(
            clust_labs, cluster_labels, max_colors
        )

    # Remap to contiguous 0..K-1
    unique_ids  = sorted(set(cluster_labels))
    remap       = {old: new for new, old in enumerate(unique_ids)}
    cluster_labels = np.array([remap[l] for l in cluster_labels])
    K           = len(unique_ids)

    # ── 5. Assign over-cap shapes to nearest cluster centroid (in LAB) ───────
    if n_shapes > n_clust:
        # Compute cluster centroids from the shapes that were clustered
        centroids = np.array([
            clust_labs[cluster_labels == k].mean(axis=0) for k in range(K)
        ])
        for i in range(n_clust, n_shapes):
            lab_i  = _rgb_to_lab(shape_rgbs[i])
            dists  = np.linalg.norm(centroids - lab_i, axis=1)
            assignment = int(np.argmin(dists))
            shape_rgbs.insert(len(clust_rgbs) + (i - n_clust), shape_rgbs[i])
            shape_pxns.insert(len(clust_pxns) + (i - n_clust), shape_pxns[i])
            cluster_labels = np.append(cluster_labels, assignment)

    # ── 6. Final ink color = pixel-count-weighted mean RGB per cluster ────────
    ink_colors: List[np.ndarray] = []
    for k in range(K):
        members = np.where(cluster_labels == k)[0]
        rgbs    = np.array([shape_rgbs[i] for i in members])
        pxns    = np.array([shape_pxns[i]  for i in members], dtype=np.float64)
        total   = pxns.sum()
        if total == 0:
            continue
        weighted = (rgbs * pxns[:, None]).sum(axis=0) / total
        ink_colors.append(weighted.astype(np.float32))

    # Sort darkest-first (smallest RGB norm = darkest)
    ink_colors.sort(key=lambda c: float(np.linalg.norm(c)))
    return ink_colors


def _merge_closest_clusters(
    lab_points: np.ndarray,   # (N, 3)
    labels:     np.ndarray,   # (N,) int
    target_k:   int,
) -> np.ndarray:
    """
    Greedily merge the two clusters with the closest average-linkage distance
    until at most *target_k* clusters remain.
    """
    labels = labels.copy()
    while len(set(labels)) > target_k:
        unique    = list(set(labels))
        centroids = {
            lbl: lab_points[labels == lbl].mean(axis=0) for lbl in unique
        }
        best_d, merge_a, merge_b = float("inf"), unique[0], unique[1]
        for i, a in enumerate(unique):
            for b in unique[i + 1:]:
                d = float(np.linalg.norm(centroids[a] - centroids[b]))
                if d < best_d:
                    best_d, merge_a, merge_b = d, a, b
        labels[labels == merge_b] = merge_a
    return labels


# ---------------------------------------------------------------------------
# Alpha unmixing — solve α for each pixel given one ink color
# ---------------------------------------------------------------------------

def _compute_ink_alpha(
    rgb_float: np.ndarray,   # (H, W, 3) float32
    paper:     np.ndarray,   # (3,) float32
    ink:       np.ndarray,   # (3,) float32
) -> np.ndarray:             # (H, W) float32 in [0, 1]
    """
    Solve the alpha-blend model per pixel:
        pixel = α·ink + (1-α)·paper
        α     = (pixel - paper)·v / (v·v),   v = ink - paper
    Returns α clipped to [0, 1].
    """
    v     = ink - paper
    denom = float(np.dot(v, v)) + 1e-6
    proj  = np.tensordot(rgb_float - paper, v, axes=([2], [0])) / denom
    return np.clip(proj, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Potrace vectorization
# ---------------------------------------------------------------------------

def _potrace_available() -> bool:
    try:
        subprocess.run(
            ["potrace", "--version"],
            capture_output=True,
            check=True,
            timeout=5,
        )
        return True
    except Exception:
        return False


_HAVE_POTRACE: bool = _potrace_available()


def _vectorize_with_potrace(
    alpha_masked: np.ndarray,           # (H, W) float32, 0–1
    ink_rgb:      Tuple[int, int, int],
    workdir:      str,
    k:            int,
) -> Optional[str]:
    """
    Vectorize an alpha mask with potrace.

    The alpha mask is thresholded at _ALPHA_THRESHOLD, inverted (potrace traces
    dark areas), and written as a PGM.  Potrace produces SVG paths; the fill
    is recoloured from black to *ink_rgb*.

    Returns the path to the SVG file, or None on any failure.
    """
    binary  = (alpha_masked > _ALPHA_THRESHOLD)
    # Potrace traces dark (0) pixels.  Ink=True → 0 (dark).
    pgm_arr = np.where(binary, 0, 255).astype(np.uint8)

    pgm_path = os.path.join(workdir, f"layer_{k}.pgm")
    svg_path = os.path.join(workdir, f"layer_{k}.svg")
    Image.fromarray(pgm_arr, "L").save(pgm_path)

    try:
        result = subprocess.run(
            [
                "potrace", "--svg",
                "--turdsize", "2",          # drop pixel-sized noise blobs
                "--output", svg_path,
                pgm_path,
            ],
            capture_output=True,
            timeout=120,
        )
    except Exception:
        return None

    if result.returncode != 0 or not os.path.exists(svg_path):
        return None

    # Re-colour paths from potrace's default black to the actual ink colour.
    r, g, b    = ink_rgb
    color_hex  = f"#{r:02x}{g:02x}{b:02x}"
    svg_text   = Path(svg_path).read_text()
    svg_text   = re.sub(
        r'fill="#[0-9a-fA-F]{6}"', f'fill="{color_hex}"', svg_text
    )
    # Remove any white/paper background rectangle that potrace might insert.
    svg_text   = re.sub(
        r'<rect[^>]*fill="#(?:ff){3}"[^>]*/>', "", svg_text, flags=re.I
    )
    Path(svg_path).write_text(svg_text)
    return svg_path


# ---------------------------------------------------------------------------
# Layer creation — alpha unmix → RGBA or vector
# ---------------------------------------------------------------------------

def _create_layers(
    rgb_array:    np.ndarray,
    paper_color:  np.ndarray,
    layer_masks:  List[np.ndarray],
    ink_colors:   List[np.ndarray],
    workdir:      str,
) -> List[_Layer]:
    """
    Build one _Layer per ink color.

    For each ink:
      1. Solve α per pixel via the alpha-blend model.
      2. Mask α to only the pixels assigned to this ink (winner region).
      3. If potrace is installed: vectorize the thresholded alpha mask → SVG.
      4. Otherwise: build an RGBA image (flat ink color at solved α, transparent bg).
    """
    rgb_float = rgb_array.astype(np.float32)
    layers: List[_Layer] = []

    for k, (mask, ink) in enumerate(zip(layer_masks, ink_colors)):
        ink_rgb = (int(ink[0]), int(ink[1]), int(ink[2]))
        name    = "layer_{}_rgb_{}_{}_{}".format(k + 1, *ink_rgb)

        # Solve α for this ink, then zero it outside the winner region.
        alpha         = _compute_ink_alpha(rgb_float, paper_color, ink)
        alpha_masked  = (alpha * mask).astype(np.float32)

        if _HAVE_POTRACE:
            svg_path = _vectorize_with_potrace(alpha_masked, ink_rgb, workdir, k + 1)
            if svg_path:
                layers.append(_Layer(
                    name     = name,
                    color    = ink.astype(np.uint8),
                    svg_path = svg_path,
                ))
                continue

        # Fallback: RGBA raster — flat ink color at solved alpha, transparent bg.
        H, W   = rgb_array.shape[:2]
        rgba   = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[..., 0] = ink_rgb[0]
        rgba[..., 1] = ink_rgb[1]
        rgba[..., 2] = ink_rgb[2]
        rgba[..., 3] = np.clip(alpha_masked * 255, 0, 255).astype(np.uint8)

        layers.append(_Layer(
            name  = name,
            color = ink.astype(np.uint8),
            image = Image.fromarray(rgba, "RGBA"),
        ))

    return layers


# ---------------------------------------------------------------------------
# Composite on forced-white background
# ---------------------------------------------------------------------------

def _build_composite_on_white(
    rgb_array:   np.ndarray,
    paper_color: np.ndarray,
    layer_masks: List[np.ndarray],
    ink_colors:  List[np.ndarray],
) -> Image.Image:
    """
    Build the composite preview on a FORCED-WHITE background.

    The user's design may be uploaded with any background color (yellow, green,
    blue, cream, etc.). Regardless of that input, the composite preview and
    every layer page must render on pure white so the separation output is
    press-ready.

    Method:
      • Start with a pure-white RGB canvas (255, 255, 255).
      • For each detected ink, solve α via the alpha-blend model on the input,
        restrict α to the winner region for that ink, and alpha-blend the flat
        ink color onto the running canvas:
              out = α · ink + (1 − α) · out
        Darkest inks first (list is already sorted that way) so lighter inks
        overlay correctly at edges.
    """
    H, W    = rgb_array.shape[:2]
    canvas  = np.full((H, W, 3), 255.0, dtype=np.float32)
    rgbf    = rgb_array.astype(np.float32)

    for mask, ink in zip(layer_masks, ink_colors):
        alpha        = _compute_ink_alpha(rgbf, paper_color, ink)
        alpha_masked = (alpha * mask).astype(np.float32)[..., None]  # (H,W,1)
        ink_arr      = ink.astype(np.float32).reshape(1, 1, 3)
        canvas       = alpha_masked * ink_arr + (1.0 - alpha_masked) * canvas

    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), "RGB")


# ---------------------------------------------------------------------------
# PDF assembly
# ---------------------------------------------------------------------------

def _resolve_page_dimensions(
    canvas_size: Tuple[int, int],
    page_size:   str,
    orientation: str,
) -> Tuple[float, float]:
    src_w, src_h       = canvas_size
    src_is_landscape   = src_w > src_h

    if page_size == "auto":
        long_side_in = 10.0
        if src_is_landscape:
            page_w = long_side_in * inch
            page_h = long_side_in * inch * (src_h / src_w)
        else:
            page_h = long_side_in * inch
            page_w = long_side_in * inch * (src_w / src_h)
    else:
        page_w, page_h = _NAMED_PAGE_SIZES[page_size]

    if orientation == "auto":
        if page_size != "auto" and src_is_landscape and page_h > page_w:
            page_w, page_h = page_h, page_w
    elif orientation == "landscape" and page_h > page_w:
        page_w, page_h = page_h, page_w
    elif orientation == "portrait" and page_w > page_h:
        page_w, page_h = page_h, page_w

    return page_w, page_h


def _build_pdf(
    layers:          List[_Layer],
    composite_image: Image.Image,
    canvas_size:     Tuple[int, int],
    page_size:       str = "auto",
    orientation:     str = "auto",
) -> bytes:
    """
    Assemble the final PDF.

    Each ink-layer page draws only the flat ink color at the solved alpha
    (vector SVG if potrace produced one, otherwise RGBA PNG with transparency).
    The final page is the original composite raster.
    All pages share the same registration frame.
    """
    page_w, page_h = _resolve_page_dimensions(canvas_size, page_size, orientation)

    short_dim    = min(page_w, page_h)
    margin       = _MARGIN_FRAC       * short_dim
    inner_pad    = _INNER_PAD_FRAC    * short_dim
    bracket_arm  = _BRACKET_ARM_FRAC  * short_dim
    crosshair_arm= _CROSSHAIR_ARM_FRAC* short_dim
    line_weight  = _LINE_WEIGHT_FRAC  * short_dim

    fx0, fy0 = margin, margin
    fx1, fy1 = page_w - margin, page_h - margin
    ax0, ay0 = fx0 + inner_pad, fy0 + inner_pad
    ax1, ay1 = fx1 - inner_pad, fy1 - inner_pad
    art_w, art_h = ax1 - ax0, ay1 - ay0

    def draw_reg_marks(c: pdf_canvas.Canvas) -> None:
        c.setStrokeColorRGB(0, 0, 0)
        c.setLineWidth(line_weight)
        for x, y, sx, sy in [
            (fx0, fy0, +1, +1), (fx1, fy0, -1, +1),
            (fx0, fy1, +1, -1), (fx1, fy1, -1, -1),
        ]:
            c.line(x, y, x + sx * bracket_arm, y)
            c.line(x, y, x, y + sy * bracket_arm)
        mx, my = (fx0 + fx1) / 2, (fy0 + fy1) / 2
        for x, y in [(mx, fy0), (mx, fy1), (fx0, my), (fx1, my)]:
            c.line(x - crosshair_arm, y, x + crosshair_arm, y)
            c.line(x, y - crosshair_arm, x, y + crosshair_arm)
            c.circle(x, y, crosshair_arm * 0.55, stroke=1, fill=0)

    def art_bounds(native_w: float, native_h: float) -> Tuple[float, float, float, float]:
        """Return (x, y, draw_w, draw_h) centred in the artwork area."""
        scale  = min(art_w / native_w, art_h / native_h)
        draw_w = native_w * scale
        draw_h = native_h * scale
        x      = ax0 + (art_w - draw_w) / 2
        y      = ay0 + (art_h - draw_h) / 2
        return x, y, draw_w, draw_h

    def draw_svg_layer(c: pdf_canvas.Canvas, svg_file: str) -> None:
        drawing = svg2rlg(svg_file)
        if drawing is None or drawing.width == 0 or drawing.height == 0:
            return
        x, y, draw_w, draw_h = art_bounds(drawing.width, drawing.height)
        scale = draw_w / drawing.width
        c.saveState()
        c.translate(x, y)
        c.scale(scale, scale)
        renderPDF.draw(drawing, c, 0, 0)
        c.restoreState()

    def draw_image_layer(
        c:   pdf_canvas.Canvas,
        img: Image.Image,
        *,
        transparent: bool = False,
    ) -> None:
        iw, ih = img.size
        x, y, draw_w, draw_h = art_bounds(iw, ih)
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        kwargs = {"mask": "auto"} if transparent else {}
        c.drawImage(ImageReader(buf), x, y, draw_w, draw_h, **kwargs)

    def draw_layer(c: pdf_canvas.Canvas, layer: _Layer) -> None:
        if layer.svg_path:
            draw_svg_layer(c, layer.svg_path)
        elif layer.image:
            draw_image_layer(c, layer.image, transparent=True)

    out = BytesIO()
    c   = pdf_canvas.Canvas(out, pagesize=(page_w, page_h))

    for layer in layers:
        draw_reg_marks(c)
        draw_layer(c, layer)
        c.showPage()

    # Composite: original artwork, all colors together.
    draw_reg_marks(c)
    draw_image_layer(c, composite_image, transparent=False)
    c.showPage()

    c.save()
    return out.getvalue()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Position screen-print separator — one artwork in, print-ready PDF out."
    )
    parser.add_argument("input",  help="Input artwork (PNG / JPG / PDF / SVG)")
    parser.add_argument("output", help="Output PDF path")
    parser.add_argument("--max-colors",  type=int, default=4)
    parser.add_argument("--page-size",   default="auto")
    parser.add_argument("--orientation", default="auto")
    args = parser.parse_args()

    generate_separation_pdf(
        args.input, args.output,
        max_colors  = args.max_colors,
        page_size   = args.page_size,
        orientation = args.orientation,
    )
    print(f"Wrote: {args.output}")
