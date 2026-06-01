"""
ratio_viz.py — spectral ratio visualisation for cam* multispectral images

Modes:

  Raw image mode (default):
    Load a raw 1×4 multispectral strip, split it on-the-fly, and visualise.

      python ratio_viz.py camMS-1234.jpg

  Aligned directory mode (single --aligned-dir):
    Load pre-split, pre-aligned per-band files produced by align_multispec.py,
    group them by MS image stem, and visualise each group.

      python ratio_viz.py --aligned-dir /path/to/aligned/
      python ratio_viz.py --aligned-dir /path/to/aligned/ -o /path/to/viz_out/
      python ratio_viz.py --aligned-dir /path/to/aligned/ --stem camMS-1234567890

    Add --summary to replace per-image saves with a single time-series plot of
    mean ratios over time (timestamp parsed from camMS-{epoch_us} stem):

      python ratio_viz.py --aligned-dir /path/to/aligned/ --summary

  Compare mode (multiple --aligned-dir):
    Provide two or more aligned directories to overlay their ratio time-series
    and compare bed/gap patterns across datasets (e.g. different fields or runs).

      python ratio_viz.py --aligned-dir /path/to/run1/ /path/to/run2/
      python ratio_viz.py --aligned-dir /path/to/run1/ /path/to/run2/ -o /path/to/out/

    Produces viz_comparison.png (and optionally ratios_<label>.json per dataset).
    X-axis is elapsed seconds from the first frame of each dataset so captures of
    different durations or starting times can be overlaid meaningfully.

Band order (all modes):
  0 → 685 nm   1 → 725 nm   2 → 750 nm   3 → 1000 nm

Note on black-pixel masking:
  The alignment pipeline can leave fully-black border regions in the aligned
  (denominator) bands.  All mean-ratio calculations automatically exclude pixels
  where the relevant denominator band is zero to prevent those artefacts from
  inflating the reported means.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


_IMG_EXTS = [".jpg", ".jpeg", ".tif", ".tiff", ".png"]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load: {path}")
    return img


def _find_band_file(directory: Path, stem: str, suffix: str) -> Path | None:
    """Return the first matching file for <stem><suffix>.<ext>, or None."""
    for ext in _IMG_EXTS:
        p = directory / f"{stem}{suffix}{ext}"
        if p.exists():
            return p
    return None


# ── Timestamp helpers ────────────────────────────────────────────────────────

def _stem_to_datetime(stem: str) -> datetime | None:
    """Parse a camMS-{epoch_us} stem into a UTC datetime, or None."""
    if not stem.startswith("camMS-"):
        return None
    try:
        epoch_us = int(stem[len("camMS-"):])
        return datetime.fromtimestamp(epoch_us / 1_000_000, tz=timezone.utc)
    except ValueError:
        return None


# ── Aligned-directory helpers ─────────────────────────────────────────────────

def find_base_stems(aligned_dir: Path) -> list[str]:
    """Return sorted base stems of MS images present in an align_multispec output dir.

    A base stem is the original MS image name with no _band* or _aligned suffix.
    Band files (_band0_1_aligned, etc.) and temp files (_avg_ff_*) are excluded.
    """
    stems: set[str] = set()
    for ext in _IMG_EXTS:
        for p in aligned_dir.glob(f"*{ext}"):
            name = p.stem
            if "_band" in name or name.startswith("_") or name.startswith("viz_"):
                continue
            # Strip _aligned suffix when RGB reference was used
            if name.endswith("_aligned"):
                name = name[: -len("_aligned")]
            stems.add(name)
    return sorted(stems)


def load_aligned_bands(
    aligned_dir: Path, base_stem: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Load the four cam* bands for *base_stem* from an align_multispec output dir.

    Band 0 is either ``{stem}.ext`` (MS reference run) or
    ``{stem}_aligned.ext`` (RGB reference run).
    Bands 1–3 are always ``{stem}_band0_{c}_aligned.ext``.

    Returns (b0, b1, b2, b3) or None if any band is missing.
    """
    band0_path = (
        _find_band_file(aligned_dir, base_stem, "")
        or _find_band_file(aligned_dir, base_stem, "_aligned")
    )
    if band0_path is None:
        return None

    bands: list[np.ndarray] = [_load_gray(band0_path)]
    for c in range(1, 4):
        p = _find_band_file(aligned_dir, base_stem, f"_band0_{c}_aligned")
        if p is None:
            return None
        bands.append(_load_gray(p))

    return bands[0], bands[1], bands[2], bands[3]


# ── Ratio computation ─────────────────────────────────────────────────────────

def _compute_ratios(
    b0: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    b3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f0 = b0.astype(np.float32)
    f1 = b1.astype(np.float32) + 1e-5
    f2 = b2.astype(np.float32) + 1e-5
    f3 = b3.astype(np.float32) + 1e-5
    return (
        np.clip(f0 / f1, 0, 1.5),
        np.clip(f0 / f2, 0, 1.5),
        np.clip(f0 / f3, 0, 1.5),
    )


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualize_bands(
    b0: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    b3: np.ndarray,
    title: str = "Multispectral Band Analysis & 685nm Ratios",
    save_path: Path | None = None,
) -> None:
    ratio_1, ratio_2, ratio_3 = _compute_ratios(b0, b1, b2, b3)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle(title, fontsize=16)

    for ax, band, label in zip(
        axes[0],
        [b0, b1, b2, b3],
        ["Band 0 (685 nm)", "Band 1 (725 nm)", "Band 2 (750 nm)", "Band 3 (1000 nm)"],
    ):
        ax.imshow(band, cmap="gray")
        ax.set_title(label)

    axes[1, 0].imshow(b0, cmap="magma")
    axes[1, 0].set_title("Baseline (685 nm)")

    for ax, ratio, label in zip(
        axes[1, 1:],
        [ratio_1, ratio_2, ratio_3],
        ["685 / Band 1", "685 / Band 2", "685 / Band 3"],
    ):
        im = ax.imshow(ratio, cmap="magma")
        ax.set_title(f"Ratio: {label}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    out = save_path or Path("viz_plot.png")
    plt.savefig(str(out))
    plt.close(fig)
    print(f"Saved: {out}")


# ── Agricultural ratio computation ────────────────────────────────────────────

def _compute_mean_ratios(
    b0: np.ndarray, b1: np.ndarray, b2: np.ndarray, b3: np.ndarray
) -> dict[str, float]:
    """Return a dict of mean per-image agricultural ratios.

    Bands: b0=685 nm  b1=725 nm  b2=750 nm  b3=1000 nm

    Ratios:
      685_725, 685_750, 685_1000 — simple band ratios (clipped 0–1.5)
      ndvi        — (750−685)/(750+685), proxy for standard NDVI
      water_index — 750/1000, proxy for plant water content
                    (1000 nm is a water absorption band; higher = more water)

    Pixels where the relevant denominator band is fully black (==0) are excluded
    from each mean to avoid alignment-border artefacts inflating the values.
    """
    f0 = b0.astype(np.float32)
    f1 = b1.astype(np.float32)
    f2 = b2.astype(np.float32)
    f3 = b3.astype(np.float32)

    # Per-denominator validity masks — exclude alignment black-border pixels
    m1 = f1 > 0
    m2 = f2 > 0
    m3 = f3 > 0

    r_685_725  = np.where(m1, np.clip(f0 / (f1 + 1e-5), 0.0, 1.5), np.nan)
    r_685_750  = np.where(m2, np.clip(f0 / (f2 + 1e-5), 0.0, 1.5), np.nan)
    r_685_1000 = np.where(m3, np.clip(f0 / (f3 + 1e-5), 0.0, 1.5), np.nan)
    ndvi       = np.where(m2, np.clip((f2 - f0) / (f2 + f0 + 1e-5), -1.0, 1.0), np.nan)
    water      = np.where(m2 & m3, np.clip(f2 / (f3 + 1e-5), 0.0, 3.0), np.nan)

    return {
        "685_725":     float(np.nanmean(r_685_725)),
        "685_750":     float(np.nanmean(r_685_750)),
        "685_1000":    float(np.nanmean(r_685_1000)),
        "ndvi":        float(np.nanmean(ndvi)),
        "water_index": float(np.nanmean(water)),
    }


# ── Dataset record loading ────────────────────────────────────────────────────

def load_dataset_records(
    aligned_dir: Path,
    stems: list[str],
) -> list[tuple[datetime, str, dict[str, float]]]:
    """Load timestamped ratio records for *stems* from *aligned_dir*.

    Stems without a parseable camMS-{epoch_us} timestamp are skipped with a
    warning.  Returns records sorted by timestamp.
    """
    records: list[tuple[datetime, str, dict[str, float]]] = []
    for stem in stems:
        bands = load_aligned_bands(aligned_dir, stem)
        if bands is None:
            print(f"  WARNING: could not load all 4 bands for '{stem}' — skipping")
            continue
        ts = _stem_to_datetime(stem)
        if ts is None:
            print(f"  WARNING: no timestamp in stem '{stem}' — excluded from time axis")
            continue
        records.append((ts, stem, _compute_mean_ratios(*bands)))
    records.sort(key=lambda x: x[0])
    return records


# ── Gap / bed-boundary detection ──────────────────────────────────────────────

def _local_maxima(values: list[float]) -> list[int]:
    """Return indices of strict local maxima (higher than both neighbours)."""
    return [
        i for i in range(1, len(values) - 1)
        if values[i] > values[i - 1] and values[i] > values[i + 1]
    ]


def _suppress_close_peaks(candidates: list[int], values: list[float], min_spacing: int) -> list[int]:
    """Greedy temporal suppression: keep a candidate only if it is at least
    *min_spacing* frames away from the last accepted peak."""
    accepted: list[int] = []
    for idx in candidates:          # candidates are already sorted by index
        if not accepted or idx - accepted[-1] >= min_spacing:
            accepted.append(idx)
    return accepted


def detect_gaps(
    records: list[tuple[datetime, str, dict[str, float]]],
    bed_n_std: float = 1.5,
    gap_n_std: float = 0.5,
    min_bed_spacing: int = 5,
) -> tuple[list[int], list[int]]:
    """Return (bed_boundary_indices, within_bed_gap_indices).

    Bed boundaries: local maxima in 685/1000 exceeding mean + bed_n_std * std,
                    then filtered so that no two accepted boundaries are within
                    *min_bed_spacing* frames of each other.
    Within-bed gaps: local maxima in 685/725 exceeding mean + gap_n_std * std
                     that don't coincide with a bed boundary.
    Returns empty lists when there are fewer than 3 records.
    """
    if len(records) < 3:
        return [], []

    vals_1000 = [r[2]["685_1000"] for r in records]
    vals_725  = [r[2]["685_725"]  for r in records]

    mean_1000, std_1000 = float(np.mean(vals_1000)), float(np.std(vals_1000))
    mean_725,  std_725  = float(np.mean(vals_725)),  float(np.std(vals_725))

    bed_thresh = mean_1000 + bed_n_std * std_1000
    gap_thresh = mean_725  + gap_n_std * std_725

    bed_candidates = sorted(i for i in _local_maxima(vals_1000) if vals_1000[i] >= bed_thresh)
    bed_idxs = _suppress_close_peaks(bed_candidates, vals_1000, min_bed_spacing)

    bed_set = set(bed_idxs)
    gap_idxs = [
        i for i in _local_maxima(vals_725)
        if vals_725[i] >= gap_thresh and i not in bed_set
    ]

    return bed_idxs, gap_idxs


# ── Summary time-series plot ──────────────────────────────────────────────────

def plot_ratio_summary(
    records: list[tuple[datetime, str, dict[str, float]]],
    save_path: Path,
    bed_idxs: list[int] | None = None,
    gap_idxs: list[int] | None = None,
) -> None:
    """Plot mean ratios over time, optionally annotating detected gaps.

    *records* is a list of (timestamp, stem, ratios_dict) sorted by time.
    """
    times = [r[0] for r in records]
    r_685_725  = [r[2]["685_725"]     for r in records]
    r_685_750  = [r[2]["685_750"]     for r in records]
    r_685_1000 = [r[2]["685_1000"]    for r in records]
    ndvi       = [r[2]["ndvi"]        for r in records]
    water      = [r[2]["water_index"] for r in records]

    fig, (ax_ratio, ax_agri) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Spectral Ratios over Time", fontsize=14)

    # — Band ratios —
    ax_ratio.plot(times, r_685_725,  marker="o", ms=4, label="685 / 725 nm")
    ax_ratio.plot(times, r_685_750,  marker="s", ms=4, label="685 / 750 nm")
    ax_ratio.plot(times, r_685_1000, marker="^", ms=4, label="685 / 1000 nm")
    ax_ratio.set_ylabel("Mean ratio (clipped 0–1.5)")
    ax_ratio.legend(fontsize=9)
    ax_ratio.grid(True, alpha=0.3)

    # — Agricultural indices —
    ax_agri.plot(times, ndvi,  marker="o", ms=4, color="green",  label="NDVI (750/685 proxy)")
    ax_agri.plot(times, water, marker="s", ms=4, color="steelblue", label="Water index (750/1000)")
    ax_agri.set_ylabel("Index value")
    ax_agri.set_xlabel("Time (UTC)")
    ax_agri.legend(fontsize=9)
    ax_agri.grid(True, alpha=0.3)

    # — Gap annotations (both panels) —
    for ax in (ax_ratio, ax_agri):
        for i in (bed_idxs or []):
            ax.axvline(times[i], color="red",    linestyle="--", linewidth=1.2, alpha=0.7,
                       label="_bed boundary" if ax is ax_ratio else "_")
        for i in (gap_idxs or []):
            ax.axvline(times[i], color="orange", linestyle=":",  linewidth=1.0, alpha=0.6,
                       label="_within-bed gap" if ax is ax_ratio else "_")

    if bed_idxs or gap_idxs:
        from matplotlib.lines import Line2D
        legend_handles = ax_ratio.get_legend_handles_labels()[0]
        extra = [
            Line2D([0], [0], color="red",    linestyle="--", label="Bed boundary"),
            Line2D([0], [0], color="orange", linestyle=":",  label="Within-bed gap"),
        ]
        ax_ratio.legend(handles=legend_handles + extra, fontsize=9)

    ax_ratio.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(str(save_path))
    plt.close(fig)
    print(f"Saved summary plot: {save_path}")


# ── Multi-dataset comparison ──────────────────────────────────────────────────

#: (dataset_label, records, bed_idxs, gap_idxs)
_DatasetEntry = tuple[str, list[tuple[datetime, str, dict[str, float]]], list[int], list[int]]


def plot_comparison(
    datasets: list[_DatasetEntry],
    save_path: Path,
) -> None:
    """Overlay ratio time-series from multiple datasets for cross-capture comparison.

    Each dataset's X-axis is elapsed seconds from its own first frame so that
    captures started at different times or on different days are still visually
    comparable.  Bed boundaries are drawn as dashed vertical lines and within-bed
    gaps as dotted lines, both in the dataset's own colour.
    """
    metrics: list[tuple[str, str]] = [
        ("685_1000", "685 / 1000 nm  (bed-gap signal, clipped 0–1.5)"),
        ("ndvi",     "NDVI  (750/685 proxy, −1 – 1)"),
        ("685_725",  "685 / 725 nm  (within-bed gap signal, clipped 0–1.5)"),
        ("water_index", "Water index  (750/1000, clipped 0–3)"),
    ]

    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=False)
    if n == 1:
        axes = [axes]
    fig.suptitle("Cross-dataset Spectral Ratio Comparison", fontsize=14)

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for ax, (key, ylabel) in zip(axes, metrics):
        for (label, records, bed_idxs, gap_idxs), color in zip(datasets, colors):
            if not records:
                continue
            t0 = records[0][0]
            rel_s = [(r[0] - t0).total_seconds() for r in records]
            vals  = [r[2][key] for r in records]
            ax.plot(rel_s, vals, marker="o", ms=3, linewidth=1.2,
                    color=color, label=label)
            for i in bed_idxs:
                ax.axvline(rel_s[i], color=color, linestyle="--",
                           linewidth=1.0, alpha=0.75)
            for i in gap_idxs:
                ax.axvline(rel_s[i], color=color, linestyle=":",
                           linewidth=0.8, alpha=0.55)

        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Elapsed time (s)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Append line-style legend entries to the first panel
    from matplotlib.lines import Line2D
    extra = [
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.0,
               label="Bed boundary"),
        Line2D([0], [0], color="gray", linestyle=":",  linewidth=0.8,
               label="Within-bed gap"),
    ]
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=handles + extra, fontsize=9)

    plt.tight_layout()
    plt.savefig(str(save_path))
    plt.close(fig)
    print(f"Saved comparison plot: {save_path}")


def run_compare(
    aligned_dirs: list[Path],
    save_dir: Path | None,
    min_bed_spacing: int = 5,
    export_json: bool = False,
) -> None:
    """Load records from multiple aligned directories and produce a comparison plot."""
    out_dir = save_dir or aligned_dirs[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets: list[_DatasetEntry] = []
    source_dirs: list[Path] = []

    for d in aligned_dirs:
        label = d.name
        stems = find_base_stems(d)
        if not stems:
            print(f"No recognisable MS image groups found in: {d}")
            continue
        print(f"[{label}] Found {len(stems)} image group(s).")
        records = load_dataset_records(d, stems)
        if not records:
            print(f"[{label}] No timestamped records — skipping.")
            continue
        bed_idxs, gap_idxs = detect_gaps(records, min_bed_spacing=min_bed_spacing)
        print(
            f"[{label}] Detected {len(bed_idxs)} bed boundary/ies "
            f"and {len(gap_idxs)} within-bed gap(s)."
        )
        datasets.append((label, records, bed_idxs, gap_idxs))
        source_dirs.append(d)

    if not datasets:
        print("No valid datasets to compare.")
        return

    plot_comparison(datasets, out_dir / "viz_comparison.png")

    if export_json:
        for (label, records, bed_idxs, gap_idxs), src in zip(datasets, source_dirs):
            export_json_fn(
                records, bed_idxs, gap_idxs, src,
                out_dir / f"ratios_{label}.json",
            )


# ── JSON export ───────────────────────────────────────────────────────────────

def export_json_fn(
    records: list[tuple[datetime, str, dict[str, float]]],
    bed_idxs: list[int],
    gap_idxs: list[int],
    source_dir: Path,
    save_path: Path,
) -> None:
    """Write per-image ratios, gap analysis, and summary statistics to JSON."""
    import json

    def _rec(i: int) -> dict:
        ts, stem, ratios = records[i]
        return {"stem": stem, "timestamp_utc": ts.isoformat(), "ratios": ratios}

    all_ratios_by_key = {
        key: [r[2][key] for r in records]
        for key in records[0][2]
    }

    data = {
        "generated": datetime.now(tz=timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "n_images": len(records),
        "per_image": [_rec(i) for i in range(len(records))],
        "gap_analysis": {
            "bed_boundaries":  [_rec(i) for i in bed_idxs],
            "within_bed_gaps": [_rec(i) for i in gap_idxs],
            "notes": {
                "bed_boundaries":  "Local maxima in 685/1000 above mean + 1.5 SD",
                "within_bed_gaps": "Local maxima in 685/725  above mean + 0.5 SD, "
                                   "not coinciding with a bed boundary",
            },
        },
        "summary": {
            "n_bed_boundaries":  len(bed_idxs),
            "n_within_bed_gaps": len(gap_idxs),
            "mean_ratios": {k: float(np.mean(v)) for k, v in all_ratios_by_key.items()},
            "std_ratios":  {k: float(np.std(v))  for k, v in all_ratios_by_key.items()},
        },
    }

    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON: {save_path}")


# ── Mode implementations ──────────────────────────────────────────────────────

def run_raw(image_path: Path, save_path: Path | None) -> None:
    """Split a raw 1×4 MS strip and visualise."""
    raw = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Could not load: {image_path}")

    h, w = raw.shape
    wb = w // 4
    b0 = raw[:, 0:wb]
    b1 = raw[:, wb : wb * 2]
    b2 = cv2.rotate(raw[:, wb * 2 : wb * 3], cv2.ROTATE_180)
    b3 = cv2.rotate(raw[:, wb * 3 : wb * 4], cv2.ROTATE_180)

    visualize_bands(
        b0, b1, b2, b3,
        title=f"Spectral Ratios — {image_path.name}",
        save_path=save_path or Path("viz_plot.png"),
    )


def run_aligned_dir(
    aligned_dir: Path,
    save_dir: Path | None,
    stem_filter: str | None,
    summary: bool = False,
    export_json: bool = False,
    min_bed_spacing: int = 5,
) -> None:
    """Load aligned bands from an align_multispec output dir and visualise.

    With *summary=True*, skip per-image saves and produce a single annotated
    time-series plot of mean + agricultural ratios, with detected gap markers.
    With *export_json=True*, write all ratios and gap analysis to ratios.json.
    """
    save_dir = save_dir or aligned_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    stems = [stem_filter] if stem_filter else find_base_stems(aligned_dir)
    if not stems:
        print(f"No recognisable MS image groups found in: {aligned_dir}")
        return

    print(f"Found {len(stems)} image group(s).")

    if summary:
        records = load_dataset_records(aligned_dir, stems)

        if not records:
            print("No timestamped records to plot.")
            return
        bed_idxs, gap_idxs = detect_gaps(records, min_bed_spacing=min_bed_spacing)

        if bed_idxs or gap_idxs:
            print(f"  Detected {len(bed_idxs)} bed boundary/ies and "
                  f"{len(gap_idxs)} within-bed gap(s).")

        plot_ratio_summary(
            records, save_dir / "viz_summary.png",
            bed_idxs=bed_idxs, gap_idxs=gap_idxs,
        )

        if export_json:
            export_json_fn(records, bed_idxs, gap_idxs, aligned_dir,
                           save_dir / "ratios.json")
    else:
        records_nosummary: list[tuple[datetime, str, dict[str, float]]] = []
        for stem in stems:
            bands = load_aligned_bands(aligned_dir, stem)
            if bands is None:
                print(f"  WARNING: could not load all 4 bands for '{stem}' — skipping")
                continue
            b0, b1, b2, b3 = bands
            visualize_bands(
                b0, b1, b2, b3,
                title=f"Spectral Ratios — {stem}",
                save_path=save_dir / f"viz_{stem}.png",
            )
            if export_json:
                ts = _stem_to_datetime(stem)
                if ts is not None:
                    records_nosummary.append((ts, stem, _compute_mean_ratios(b0, b1, b2, b3)))

        if export_json and records_nosummary:
            records_nosummary.sort(key=lambda x: x[0])
            export_json_fn(records_nosummary, [], [], aligned_dir,
                           save_dir / "ratios.json")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise spectral band ratios for cam* multispectral images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Raw image mode
    parser.add_argument(
        "image", nargs="?", type=Path, default=None,
        help="Raw 1×4 multispectral image (raw mode)",
    )

    # Aligned directory mode (one dir → single-dataset; two+ dirs → compare mode)
    parser.add_argument(
        "--aligned-dir", type=Path, nargs="+", default=None, metavar="DIR",
        help="Output directory/ies from align_multispec.py.  Pass one directory "
             "for normal aligned mode or two+ to compare datasets side-by-side.",
    )
    parser.add_argument(
        "--stem", type=str, default=None, metavar="STEM",
        help="Only visualise this MS image stem (aligned mode only)",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Instead of per-image saves, produce a single time-series plot of "
             "mean ratios over time (aligned mode only; requires camMS-* stems)",
    )
    parser.add_argument(
        "--min-bed-spacing", type=int, default=5, metavar="N",
        help="Minimum number of frames between accepted bed boundaries "
             "(suppresses duplicate detections from the same bed edge)",
    )
    parser.add_argument(
        "--export-json", action="store_true",
        help="Save per-image ratios (685/725, 685/750, 685/1000, NDVI, water index), "
             "gap analysis, and summary statistics to ratios.json in the output directory "
             "(aligned mode only; gap analysis requires --summary)",
    )

    # Shared
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None, metavar="DIR",
        help="Directory to save visualisation PNGs "
             "[raw default: viz_plot.png; aligned default: same as --aligned-dir]",
    )

    args = parser.parse_args()

    if args.aligned_dir is not None:
        dirs = [d.resolve() for d in args.aligned_dir]
        for d in dirs:
            if not d.exists():
                parser.error(f"--aligned-dir not found: {d}")
        save_dir = args.output_dir.resolve() if args.output_dir else None

        if len(dirs) > 1:
            if args.stem:
                parser.error("--stem cannot be used with multiple --aligned-dir arguments")
            if args.summary:
                parser.error("--summary cannot be used with multiple --aligned-dir arguments; "
                             "compare mode always produces a single comparison plot")
            run_compare(
                aligned_dirs=dirs,
                save_dir=save_dir,
                min_bed_spacing=args.min_bed_spacing,
                export_json=args.export_json,
            )
        else:
            run_aligned_dir(
                aligned_dir=dirs[0],
                save_dir=save_dir,
                stem_filter=args.stem,
                summary=args.summary,
                export_json=args.export_json,
                min_bed_spacing=args.min_bed_spacing,
            )
    elif args.image is not None:
        if not args.image.exists():
            parser.error(f"Image not found: {args.image}")
        save_path = (args.output_dir / f"viz_{args.image.stem}.png") if args.output_dir else None
        run_raw(args.image.resolve(), save_path)
    else:
        parser.error("Provide a raw image path or --aligned-dir.")


if __name__ == "__main__":
    main()
