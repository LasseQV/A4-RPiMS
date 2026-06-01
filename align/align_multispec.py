"""
align_multispec.py — split multispectral images and align their bands

Each MS image is split into per-band files, then every non-reference band is
warped onto the reference band (or onto a matched RGB image when --rgb is given).

Filename conventions
--------------------
  camMS-{epoch_us}.*  →  RPi MS camera, 1×4 strip; epoch in microseconds
  e*.*                →  Spectral Devices MSIS, 2×2 grid

RGB reference timestamp
-----------------------
  Extracted from EXIF DateTimeOriginal (+SubSecTimeOriginal if present).
  When --rgb is a directory, each MS image is paired with the closest-in-time
  RGB image.  For camMS-* the epoch is taken from the filename; for e* images
  EXIF is attempted on the MS side too.

Usage
-----
  python align_multispec.py <ms_input> [options]

  ms_input                Single MS image or a directory of MS images.

  --rgb PATH              RGB reference image or directory.
  --ref-band INT          Band index to use as MS reference  [default: 0]
                            cam* (1×4): 0 … 3 (left → right)
                            e*  (2×2): 0=(r0,c0) 1=(r0,c1) 2=(r1,c0) 3=(r1,c1)
  -o / --output-dir PATH  Output directory  [default: <ms_dir>/aligned/]
  --max-time-diff FLOAT   Reject RGB–MS pairs farther apart than this many
                          seconds (warn-only when omitted).
  --matcher STR           vismatch model name  [default: superpoint-lightglue]
  --match-size INT        Feature-extraction resolution  [default: 512]
  --device STR            auto | cpu | cuda | mps  [default: auto]
  --show-matches          Save match-visualisation images alongside output.
  --keep-splits           Keep intermediate per-band split files.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2


# ── Timestamp helpers ─────────────────────────────────────────────────────────

def _ts_from_ms_stem(path: Path) -> float | None:
    """Parse epoch seconds from a camMS-{epoch_us} filename stem."""
    stem = path.stem
    if not stem.startswith("camMS-"):
        return None
    try:
        return int(stem[len("camMS-"):]) / 1_000_000
    except ValueError:
        return None


def _ts_from_exif(path: Path) -> float | None:
    """Read DateTimeOriginal (+SubSecTimeOriginal) from EXIF; return epoch seconds."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            exif = img._getexif()
        if exif is None:
            return None
        dt_str = exif.get(36867)   # DateTimeOriginal
        subsec = exif.get(37521)   # SubSecTimeOriginal
        if dt_str is None:
            return None
        epoch = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S").timestamp()
        if subsec:
            try:
                epoch += float(f"0.{subsec.strip()}")
            except ValueError:
                pass
        return epoch
    except Exception:
        return None


def get_ms_timestamp(path: Path) -> float | None:
    """Best-effort timestamp: filename epoch first, then EXIF."""
    ts = _ts_from_ms_stem(path)
    return ts if ts is not None else _ts_from_exif(path)


# ── Band path helpers ─────────────────────────────────────────────────────────

def _grid_shape(name: str) -> tuple[int, int]:
    if name.startswith("cam"):
        return 1, 4
    if name.startswith("e"):
        return 2, 2
    raise ValueError(f"Unrecognised MS filename prefix: {name!r}")


def band_paths_in_dir(ms_path: Path, split_dir: Path) -> list[Path]:
    """Ordered band paths (row-major) matching image_split.py naming."""
    rows, cols = _grid_shape(ms_path.name)
    paths = []
    for r in range(rows):
        for c in range(cols):
            if r == 0 and c == 0:
                fname = f"{ms_path.stem}{ms_path.suffix}"
            else:
                fname = f"{ms_path.stem}_band{r}_{c}{ms_path.suffix}"
            paths.append(split_dir / fname)
    return paths


# ── RGB directory index / MS reference lookup ────────────────────────────────

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def index_rgb_dir(rgb_dir: Path) -> list[tuple[float | None, Path]]:
    entries = []
    for p in sorted(rgb_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
            entries.append((_ts_from_exif(p), p))
    return entries


def find_ms_refs_in_dir(directory: Path) -> list[tuple[float | None, Path]]:
    """Find MS band-0 reference images in an align_multispec output directory.

    Band-0 files have an MS-style stem (camMS-* or e*) without a _band suffix.
    Both the MS-reference variant ({stem}.ext) and the RGB-reference variant
    ({stem}_aligned.ext) are matched.  Returns [(timestamp_or_None, path), ...]
    sorted by timestamp (files with no timestamp sorted to the end).
    """
    refs = []
    for p in sorted(directory.iterdir()):
        if not p.is_file() or p.suffix.lower() not in _MS_EXTS:
            continue
        stem = p.stem
        if "_band" in stem or stem.startswith("_") or stem.startswith("viz_"):
            continue
        refs.append((get_ms_timestamp(p), p))
    return sorted(refs, key=lambda x: (x[0] is None, x[0] or 0.0))


# If the best initial match exceeds this, try whole-hour UTC offsets to handle
# cameras whose EXIF clock is set to local time (e.g. PDT = UTC-7) while the
# MS system clock is UTC.
_TZ_FALLBACK_THRESHOLD = 1800  # seconds (30 minutes)
_UTC_HOUR_OFFSETS = [h * 3600 for h in range(-12, 13) if h != 0]


def find_closest_rgb(
    ms_ts: float,
    rgb_entries: list[tuple[float | None, Path]],
    max_diff: float | None,
) -> tuple[Path | None, float | None]:
    candidates = [(ts, p) for ts, p in rgb_entries if ts is not None]
    if not candidates:
        return None, None

    def _best(query: float) -> tuple[Path, float]:
        t, p = min(candidates, key=lambda x: abs(x[0] - query))
        return p, abs(t - query)

    best_path, diff = _best(ms_ts)

    if diff > _TZ_FALLBACK_THRESHOLD:
        best_offset = 0
        for offset in _UTC_HOUR_OFFSETS:
            adj_path, adj_diff = _best(ms_ts + offset)
            if adj_diff < diff:
                best_path, diff = adj_path, adj_diff
                best_offset = offset
        if best_offset != 0:
            h = best_offset // 3600
            print(f"  (tz fallback: RGB interpreted as UTC{h:+d}, Δt={diff:.1f}s)")

    if max_diff is not None and diff > max_diff:
        return None, diff
    return best_path, diff


def sorted_ts_candidates(
    query_ts: float,
    entries: list[tuple[float | None, Path]],
    max_diff: float | None,
) -> list[tuple[float, Path]]:
    """Return [(diff, path), ...] sorted by |timestamp − query_ts|, filtered by max_diff."""
    ranked = sorted(
        ((abs(ts - query_ts), p) for ts, p in entries if ts is not None),
        key=lambda x: x[0],
    )
    if max_diff is not None:
        ranked = [(d, p) for d, p in ranked if d <= max_diff]
    return ranked


# ── Timestamp-offset probe ───────────────────────────────────────────────────

def probe_ts_offset(
    ms_ref_path: Path,
    rgb_entries: list[tuple[float | None, Path]],
    matcher_name: str,
    match_size: int,
    device: str,
) -> float | None:
    """Empirically determine the clock offset between RGB EXIF and MS epoch timestamps.

    Runs feature matching between *ms_ref_path* and every timestamped RGB candidate,
    selects the one with the most inliers, and returns:

        offset = rgb_best_ts − ms_ref_ts

    Callers should adjust queries as follows before passing to find_closest_rgb:
      --align-rgb mode : find_closest_rgb(rgb_ts - offset, ms_refs, ...)
      --rgb       mode : find_closest_rgb(ms_ts  + offset, rgb_entries, ...)

    Returns None if the offset cannot be determined (missing timestamps, no matches).
    """
    ms_ts = get_ms_timestamp(ms_ref_path)
    if ms_ts is None:
        print("  WARNING: probe-ts-offset: no MS timestamp on reference image — skipped")
        return None

    candidates = [(ts, p) for ts, p in rgb_entries if ts is not None][:100]
    if not candidates:
        print("  WARNING: probe-ts-offset: no timestamped RGB candidates — skipped")
        return None

    _ensure_project_path()
    lightglue_dir = str(Path(__file__).parent / "lightGlue")
    if lightglue_dir not in sys.path:
        sys.path.insert(0, lightglue_dir)

    from vismatch import get_matcher
    from vismatch.utils import get_default_device

    _device = get_default_device() if device == "auto" else device

    print(f"\nProbing timestamp offset against {len(candidates)} RGB image(s)…")
    matcher = get_matcher(matcher_name, device=_device)
    img0 = matcher.load_image(ms_ref_path, resize=match_size)

    best_inliers = -1
    best_ts: float | None = None
    best_name = ""

    for ts, rgb_path in candidates:
        img1 = matcher.load_image(rgb_path, resize=match_size)
        result = matcher(img0, img1)
        n = result.get("num_inliers", 0)
        print(f"  {rgb_path.name}: {n} inliers")
        if n > best_inliers:
            best_inliers, best_ts, best_name = n, ts, rgb_path.name

    if best_ts is None:
        return None

    offset = best_ts - ms_ts
    h = int(offset // 3600)
    print(f"  → Best match : {best_name}  ({best_inliers} inliers)")
    print(f"  → Offset     : {offset:+.3f}s  (≈ UTC{h:+d} clock skew)")
    return offset


# ── Vignette correction helpers ───────────────────────────────────────────────

def _ensure_project_path() -> None:
    project_root = str(Path(__file__).parent.resolve())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def compute_vignette_k_values(
    ms_images: list[Path],
    splits_dir: Path,
) -> dict[int, float]:
    """Split all MS images, average each band across images, return optimal k per band.

    For a single input image the band itself serves as the flat-field estimate.
    For a directory the per-band average suppresses scene content so the vignetting
    pattern dominates.
    """
    _ensure_project_path()
    from utils.image_split import split_and_save_multiband_image
    from vignette_correction.vignette import create_average_flat_field, find_optimal_k

    splits_dir.mkdir(parents=True, exist_ok=True)

    print("\nVignette pre-pass: splitting images for flat-field estimation...")
    for ms_path in ms_images:
        split_and_save_multiband_image(ms_path, splits_dir)

    first_bands = band_paths_in_dir(ms_images[0], splits_dir)
    n_bands = len(first_bands)

    k_values: dict[int, float] = {}
    print("Computing per-band k values:")
    for band_idx in range(n_bands):
        existing = [
            p for ms in ms_images
            for p in [band_paths_in_dir(ms, splits_dir)[band_idx]]
            if p.exists()
        ]
        if not existing:
            print(f"  Band {band_idx}: no split files — k=0")
            k_values[band_idx] = 0.0
            continue

        avg_ff_path = splits_dir / f"_avg_ff_band{band_idx}.tif"
        create_average_flat_field([str(p) for p in existing], str(avg_ff_path))
        print(f"  Band {band_idx}: ", end="", flush=True)
        k_values[band_idx] = find_optimal_k(str(avg_ff_path))
        avg_ff_path.unlink(missing_ok=True)

    return k_values


def _apply_vignette_to_splits(band_paths: list[Path], k_values: dict[int, float]) -> None:
    """Apply per-band vignette correction in-place to split files."""
    _ensure_project_path()
    from vignette_correction.vignette import correct_vignette_grayscale

    for i, band_path in enumerate(band_paths):
        k = k_values.get(i, 0.0)
        if k == 0.0 or not band_path.exists():
            continue
        img = cv2.imread(str(band_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        cv2.imwrite(str(band_path), correct_vignette_grayscale(img, k))


# ── Per-image pipeline ────────────────────────────────────────────────────────

def _expected_output_paths(
    ms_path: Path,
    output_dir: Path,
    ref_band_idx: int,
    using_ms_ref: bool,
) -> list[Path]:
    """Return the output file paths that process_ms_image would produce."""
    rows, cols = _grid_shape(ms_path.name)
    paths = []
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        band_stem = ms_path.stem if (r == 0 and c == 0) else f"{ms_path.stem}_band{r}_{c}"
        if using_ms_ref and idx == ref_band_idx:
            paths.append(output_dir / f"{band_stem}{ms_path.suffix}")
        else:
            paths.append(output_dir / f"{band_stem}_aligned{ms_path.suffix}")
    return paths


def process_ms_image(
    ms_path: Path,
    output_dir: Path,
    ref_band_idx: int,
    rgb_file: Path | None,
    rgb_entries: list[tuple[float | None, Path]] | None,
    max_time_diff: float | None,
    matcher_name: str,
    match_size: int,
    device: str,
    show_matches: bool,
    keep_splits: bool,
    vignette_k: dict[int, float] | None = None,
    ts_offset: float = 0.0,
) -> None:
    print(f"\n── {ms_path.name}")

    using_ms_ref = rgb_entries is None and rgb_file is None

    expected = _expected_output_paths(ms_path, output_dir, ref_band_idx, using_ms_ref)
    if all(p.exists() for p in expected):
        print(f"  Skipping — {len(expected)} output file(s) already present")
        return

    _ensure_project_path()
    from utils.image_split import split_and_save_multiband_image
    from lightGlue.align import align_images

    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    split_and_save_multiband_image(ms_path, split_dir)
    band_paths = band_paths_in_dir(ms_path, split_dir)

    if vignette_k is not None:
        _apply_vignette_to_splits(band_paths, vignette_k)
    n_bands = len(band_paths)

    if ref_band_idx >= n_bands:
        print(f"  ERROR: --ref-band {ref_band_idx} out of range ({n_bands} bands) — skipping")
        return

    # Resolve reference image — with retry on low inliers for RGB-ref mode
    if rgb_entries is not None:
        ms_ts = get_ms_timestamp(ms_path)
        if ms_ts is None:
            print(f"  WARNING: no timestamp for {ms_path.name} — cannot match RGB, skipping")
            return
        rgb_candidates = sorted_ts_candidates(ms_ts + ts_offset, rgb_entries, max_time_diff)
        if not rgb_candidates:
            print(f"  WARNING: no RGB within {max_time_diff}s — skipping")
            return
        ref_path = rgb_candidates[0][1]
        print(f"  RGB ref : {ref_path.name}  (Δt={rgb_candidates[0][0]:.3f}s)")
    elif rgb_file is not None:
        ref_path = rgb_file
        rgb_candidates = [(0.0, rgb_file)]
        print(f"  RGB ref : {ref_path.name}")
    else:
        ref_path = band_paths[ref_band_idx]
        rgb_candidates = []
        print(f"  MS  ref : band {ref_band_idx} ({ref_path.name})")

    for i, band_path in enumerate(band_paths):
        if not band_path.exists():
            print(f"  WARNING: expected split file not found: {band_path.name}")
            continue

        if using_ms_ref and i == ref_band_idx:
            # Copy reference band to output so all results are in one place
            dest = output_dir / band_path.name
            if dest != band_path:
                shutil.copy2(band_path, dest)
            continue

        out_path = output_dir / f"{band_path.stem}_aligned{band_path.suffix}"
        print(f"  Band {i}  ({band_path.name}) → {out_path.name}")

        # For RGB-ref mode try next candidates if inliers are too low
        candidates_to_try = rgb_candidates if rgb_candidates else [(0.0, ref_path)]
        aligned = None
        for diff, cand_ref in candidates_to_try:
            if cand_ref is not ref_path:
                print(f"    Retrying with RGB {cand_ref.name}  (Δt={diff:.3f}s)")
            aligned = align_images(
                source_path=cand_ref,
                target_path=band_path,
                output_path=out_path,
                matcher_name=matcher_name,
                match_size=match_size,
                device=device,
                save_matches=show_matches,
                min_inliers=20,
            )
            if aligned is not None:
                break
            print(f"    Too few inliers — trying next candidate")
        if aligned is None:
            print(f"  WARNING: no reference yielded ≥ 20 inliers for band {i} — skipped")

    if not keep_splits:
        for bp in band_paths:
            bp.unlink(missing_ok=True)
        try:
            split_dir.rmdir()
        except OSError:
            pass


# ── Entry point ───────────────────────────────────────────────────────────────

_MS_EXTS = {".jpg", ".jpeg", ".tif", ".tiff"}


def collect_ms_images(ms_input: Path) -> list[Path]:
    if ms_input.is_file():
        return [ms_input]
    imgs: list[Path] = []
    for ext in _MS_EXTS:
        imgs.extend(ms_input.glob(f"*{ext}"))
    return sorted(set(imgs))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split multispectral images and align their bands.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("ms_input", type=Path,
                        help="Single MS image or directory of MS images. "
                             "When --align-rgb is used, treated as the MS reference source: "
                             "a single file is split on-the-fly; a directory is searched for "
                             "existing band-0 reference images (aligned-dir output).")
    parser.add_argument("--align-rgb", type=Path, default=None, metavar="PATH",
                        help="RGB image or directory to align INTO the MS reference frame. "
                             "Activates reverse mode: each RGB is warped to match the MS "
                             "band-0 coordinate frame instead of the normal MS→MS alignment.")
    parser.add_argument("--rgb", type=Path, default=None, metavar="PATH",
                        help="RGB reference image or directory (matched by timestamp)")
    parser.add_argument("--ref-band", type=int, default=0, metavar="INT",
                        help="MS band index used as reference when no --rgb is given")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="Output directory [default: <ms_dir>/aligned/]")
    parser.add_argument("--max-time-diff", type=float, default=1.0, metavar="SEC",
                        help="Reject RGB–MS pairs whose timestamps differ by more than SEC seconds "
                             "[default: 1.0; increase or omit to allow looser matching]")
    parser.add_argument("--matcher", default="superpoint-lightglue",
                        help="vismatch model name")
    parser.add_argument("--match-size", type=int, default=512,
                        help="Square resolution for feature matching")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                        help="Compute device")
    parser.add_argument("--show-matches", action="store_true",
                        help="Save match-visualisation images alongside output")
    parser.add_argument("--keep-splits", action="store_true",
                        help="Keep intermediate per-band split files in <output_dir>/splits/")
    parser.add_argument("--probe-ts-offset", action="store_true",
                        help="Before matching, run feature matching between the first MS "
                             "reference and all RGB candidates to empirically determine the "
                             "clock offset between the two cameras. The best-inlier RGB image "
                             "is used to compute a global offset applied to all subsequent "
                             "timestamp comparisons. Useful when RGB EXIF time is in local "
                             "time while MS uses UTC epoch.")
    parser.add_argument("--vignette", action="store_true",
                        help="Apply automatic vignette correction to each band before alignment. "
                             "Flat-field is estimated by averaging all bands across input images.")

    args = parser.parse_args()

    ms_input: Path = args.ms_input.resolve()
    if not ms_input.exists():
        parser.error(f"ms_input not found: {ms_input}")

    ms_dir = ms_input if ms_input.is_dir() else ms_input.parent
    output_dir: Path = args.output_dir.resolve() if args.output_dir else ms_dir / "aligned"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── RGB-to-MS alignment mode ───────────────────────────────────────────────
    if args.align_rgb is not None:
        rgb_input: Path = args.align_rgb.resolve()
        if not rgb_input.exists():
            parser.error(f"--align-rgb path not found: {rgb_input}")

        # Collect RGB images
        if rgb_input.is_file():
            rgb_images_to_align = [rgb_input]
        else:
            rgb_images_to_align = sorted(
                p for p in rgb_input.iterdir()
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
            )
        if not rgb_images_to_align:
            parser.error(f"No images found at --align-rgb path: {rgb_input}")

        # Resolve MS references
        if ms_input.is_file():
            _ensure_project_path()
            from utils.image_split import split_and_save_multiband_image
            split_dir = output_dir / "splits"
            split_dir.mkdir(parents=True, exist_ok=True)
            split_and_save_multiband_image(ms_input, split_dir)
            ref_band_path = band_paths_in_dir(ms_input, split_dir)[args.ref_band]
            if not ref_band_path.exists():
                parser.error(f"Split ref band not found for: {ms_input}")
            ms_refs: list[tuple[float | None, Path]] = [
                (get_ms_timestamp(ms_input), ref_band_path)
            ]
            print(f"MS reference : {ref_band_path.name} (split band {args.ref_band} of {ms_input.name})")
        else:
            ms_refs = find_ms_refs_in_dir(ms_input)
            if not ms_refs:
                parser.error(
                    f"No MS band-0 reference images found in: {ms_input}\n"
                    "Pass an align_multispec output directory or a single MS image."
                )
            with_ts = sum(1 for ts, _ in ms_refs if ts is not None)
            print(f"MS references: {len(ms_refs)} found in {ms_input}  ({with_ts} with timestamps)")

        _ensure_project_path()
        from lightGlue.align import align_images

        use_timestamp_match = (
            len(ms_refs) > 1
            and any(ts is not None for ts, _ in ms_refs)
        )

        # Probe for clock offset between RGB EXIF and MS epoch
        rgb_ts_offset = 0.0
        if args.probe_ts_offset and use_timestamp_match:
            rgb_probe_entries = [
                (_ts_from_exif(p), p) for p in rgb_images_to_align
            ]
            discovered = probe_ts_offset(
                ms_refs[0][1], rgb_probe_entries,
                args.matcher, args.match_size, args.device,
            )
            if discovered is not None:
                rgb_ts_offset = discovered

        print(f"RGB images   : {len(rgb_images_to_align)}")
        print(f"Output dir   : {output_dir}")

        for rgb_path in rgb_images_to_align:
            print(f"\n── {rgb_path.name}")
            if use_timestamp_match:
                rgb_ts = _ts_from_exif(rgb_path)
                if rgb_ts is None:
                    print("  WARNING: no EXIF timestamp — skipping")
                    continue
                candidates = sorted_ts_candidates(
                    rgb_ts - rgb_ts_offset, ms_refs, args.max_time_diff
                )
                if not candidates:
                    print(f"  WARNING: no MS ref within {args.max_time_diff}s — skipping")
                    continue
            else:
                candidates = [(0.0, ms_refs[0][1])]

            aligned = None
            for diff, ref_path in candidates:
                print(f"  MS ref  : {ref_path.name}  (Δt={diff:.3f}s)")
                out_stem = ref_path.stem if use_timestamp_match else rgb_path.stem
                out_path = output_dir / f"{out_stem}_aligned{rgb_path.suffix}"
                aligned = align_images(
                    source_path=ref_path,
                    target_path=rgb_path,
                    output_path=out_path,
                    matcher_name=args.matcher,
                    match_size=args.match_size,
                    device=args.device,
                    save_matches=args.show_matches,
                    min_inliers=20,
                )
                if aligned is not None:
                    break
                print(f"  Too few inliers with {ref_path.name} — trying next candidate")

            if aligned is None:
                print(f"  WARNING: no MS reference yielded ≥ 20 inliers — skipping {rgb_path.name}")

        print("\nDone.")
        return

    # ── MS band-split + alignment mode (original) ─────────────────────────────
    ms_images = collect_ms_images(ms_input)
    if not ms_images:
        parser.error(f"No MS images (.jpg/.jpeg/.tif/.tiff) found in: {ms_input}")

    rgb_file: Path | None = None
    rgb_entries: list[tuple[float | None, Path]] | None = None
    if args.rgb is not None:
        rgb = args.rgb.resolve()
        if not rgb.exists():
            parser.error(f"--rgb path not found: {rgb}")
        if rgb.is_dir():
            rgb_entries = index_rgb_dir(rgb)
            with_ts = sum(1 for ts, _ in rgb_entries if ts is not None)
            print(f"Indexed {len(rgb_entries)} RGB images  ({with_ts} with EXIF timestamps)")
        else:
            rgb_file = rgb

    print(f"Output dir : {output_dir}")
    print(f"MS images  : {len(ms_images)}")
    if not args.rgb:
        print(f"Reference  : band {args.ref_band} of each MS image")

    vignette_k: dict[int, float] | None = None
    if args.vignette:
        vignette_k = compute_vignette_k_values(ms_images, output_dir / "splits")

    # Probe for clock offset: split first MS image and match its ref-band
    # against all RGB candidates to determine a global timestamp offset.
    ms_ts_offset = 0.0
    if args.probe_ts_offset and rgb_entries:
        _ensure_project_path()
        from utils.image_split import split_and_save_multiband_image
        probe_split_dir = output_dir / "splits"
        probe_split_dir.mkdir(parents=True, exist_ok=True)
        split_and_save_multiband_image(ms_images[0], probe_split_dir)
        probe_ref = band_paths_in_dir(ms_images[0], probe_split_dir)[args.ref_band]
        if probe_ref.exists():
            discovered = probe_ts_offset(
                probe_ref, rgb_entries,
                args.matcher, args.match_size, args.device,
            )
            if discovered is not None:
                ms_ts_offset = discovered

    for ms_path in ms_images:
        process_ms_image(
            ms_path=ms_path,
            output_dir=output_dir,
            ref_band_idx=args.ref_band,
            rgb_file=rgb_file,
            rgb_entries=rgb_entries,
            max_time_diff=args.max_time_diff,
            matcher_name=args.matcher,
            match_size=args.match_size,
            device=args.device,
            show_matches=args.show_matches,
            keep_splits=args.keep_splits,
            vignette_k=vignette_k,
            ts_offset=ms_ts_offset,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
