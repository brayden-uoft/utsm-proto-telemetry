"""Convert rough inner/outer GPX boundaries into a smooth closed centerline.

The Google Earth export used for Autodrome Chaudiere contains two tracks:
an outer edge and an inner edge. This script projects both rings to local
metres, aligns their direction and start phase, averages them, applies a
periodic Gaussian smoother, and writes a geometry-only GPX at fixed spacing.
"""

from __future__ import annotations

import argparse
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


GPX_NS = "http://www.topografix.com/GPX/1/1"
EARTH_RADIUS_M = 6_371_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a smooth centerline from inner and outer GPX boundary tracks."
    )
    parser.add_argument("input", help="GPX containing at least two closed boundary tracks")
    parser.add_argument("output", help="Output geometry-only centerline GPX")
    parser.add_argument("--preview", help="Optional PNG showing raw boundaries and centerline")
    parser.add_argument("--spacing-m", type=float, default=5.0)
    parser.add_argument("--smooth-window-m", type=float, default=12.0)
    parser.add_argument("--alignment-samples", type=int, default=720)
    return parser.parse_args()


def read_boundary_tracks(path: str | os.PathLike[str]) -> list[tuple[str, np.ndarray]]:
    root = ET.parse(path).getroot()
    ns = {"gpx": GPX_NS}
    tracks: list[tuple[str, np.ndarray]] = []
    for index, track in enumerate(root.findall("gpx:trk", ns), start=1):
        name_node = track.find("gpx:name", ns)
        name = name_node.text.strip() if name_node is not None and name_node.text else f"Track {index}"
        points = []
        for point in track.findall("gpx:trkseg/gpx:trkpt", ns):
            points.append((float(point.attrib["lat"]), float(point.attrib["lon"])))
        if len(points) >= 4:
            tracks.append((name, np.asarray(points, dtype=float)))
    if len(tracks) < 2:
        raise ValueError("Input GPX must contain at least two boundary tracks.")
    return tracks


def project_local(rings: list[np.ndarray]) -> tuple[list[np.ndarray], float, float]:
    all_points = np.vstack(rings)
    lat0 = float(np.mean(all_points[:, 0]))
    lon0 = float(np.mean(all_points[:, 1]))
    cos_lat = math.cos(math.radians(lat0))
    projected = []
    for ring in rings:
        y = np.radians(ring[:, 0] - lat0) * EARTH_RADIUS_M
        x = np.radians(ring[:, 1] - lon0) * EARTH_RADIUS_M * cos_lat
        projected.append(np.column_stack((x, y)))
    return projected, lat0, lon0


def unproject_local(points: np.ndarray, lat0: float, lon0: float) -> np.ndarray:
    cos_lat = math.cos(math.radians(lat0))
    lat = lat0 + np.degrees(points[:, 1] / EARTH_RADIUS_M)
    lon = lon0 + np.degrees(points[:, 0] / (EARTH_RADIUS_M * cos_lat))
    return np.column_stack((lat, lon))


def strip_duplicate_close(points: np.ndarray) -> np.ndarray:
    if len(points) > 1 and np.linalg.norm(points[0] - points[-1]) < 0.05:
        return points[:-1]
    return points


def signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def closed_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).sum())


def resample_closed(points: np.ndarray, count: int) -> np.ndarray:
    points = strip_duplicate_close(np.asarray(points, dtype=float))
    if len(points) < 3:
        raise ValueError("A closed ring needs at least three distinct points.")
    closed = np.vstack((points, points[0]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    keep = np.concatenate(([True], lengths > 1e-6))
    closed = closed[keep]
    if len(closed) < 4:
        raise ValueError("Boundary collapses after duplicate-point removal.")
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))))
    targets = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    x = np.interp(targets, cumulative, closed[:, 0])
    y = np.interp(targets, cumulative, closed[:, 1])
    return np.column_stack((x, y))


def align_rings(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, int, bool]:
    best_error = math.inf
    best = candidate
    best_shift = 0
    best_reversed = False
    for reversed_order in (False, True):
        oriented = candidate if not reversed_order else candidate[::-1]
        for shift in range(len(oriented)):
            shifted = np.roll(oriented, shift, axis=0)
            error = float(np.mean(np.sum((reference - shifted) ** 2, axis=1)))
            if error < best_error:
                best_error = error
                best = shifted
                best_shift = shift
                best_reversed = reversed_order
    return best, best_shift, best_reversed


def circular_gaussian_smooth(points: np.ndarray, window_m: float) -> np.ndarray:
    if window_m <= 0:
        return points.copy()
    spacing = closed_length(points) / len(points)
    sigma_samples = max(window_m / max(spacing, 1e-9) / 2.355, 0.5)
    radius = max(2, int(math.ceil(3.0 * sigma_samples)))
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 * (offsets / sigma_samples) ** 2)
    weights /= weights.sum()
    smoothed = np.zeros_like(points)
    for offset, weight in zip(offsets, weights):
        smoothed += np.roll(points, int(offset), axis=0) * weight
    return smoothed


def build_centerline(
    outer: np.ndarray,
    inner: np.ndarray,
    *,
    alignment_samples: int = 720,
    smooth_window_m: float = 12.0,
    spacing_m: float = 5.0,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    if alignment_samples < 32:
        raise ValueError("--alignment-samples must be at least 32.")
    if spacing_m <= 0:
        raise ValueError("--spacing-m must be positive.")

    outer = strip_duplicate_close(outer)
    inner = strip_duplicate_close(inner)
    if signed_area(outer) * signed_area(inner) < 0:
        inner = inner[::-1]
    outer_dense = resample_closed(outer, alignment_samples)
    inner_dense = resample_closed(inner, alignment_samples)
    inner_aligned, shift, reversed_order = align_rings(outer_dense, inner_dense)
    raw_center = (outer_dense + inner_aligned) / 2.0
    smooth_center = circular_gaussian_smooth(raw_center, smooth_window_m)
    output_count = max(8, int(round(closed_length(smooth_center) / spacing_m)))
    centerline = resample_closed(smooth_center, output_count)
    metrics: dict[str, float | int | bool] = {
        "outer_length_m": closed_length(outer),
        "inner_length_m": closed_length(inner),
        "centerline_length_m": closed_length(centerline),
        "point_count": len(centerline),
        "mean_spacing_m": closed_length(centerline) / len(centerline),
        "alignment_shift": shift,
        "alignment_reversed": reversed_order,
    }
    return centerline, metrics


def write_gpx(path: str | os.PathLike[str], lat_lon: np.ndarray, name: str) -> None:
    ET.register_namespace("", GPX_NS)
    root = ET.Element(f"{{{GPX_NS}}}gpx", {"version": "1.1", "creator": "UTSM preprocess_track.py"})
    metadata = ET.SubElement(root, f"{{{GPX_NS}}}metadata")
    description = ET.SubElement(metadata, f"{{{GPX_NS}}}desc")
    description.text = "Smoothed geometry-only centerline derived from Google Earth inner and outer boundaries."
    track = ET.SubElement(root, f"{{{GPX_NS}}}trk")
    ET.SubElement(track, f"{{{GPX_NS}}}name").text = name
    segment = ET.SubElement(track, f"{{{GPX_NS}}}trkseg")
    closed = np.vstack((lat_lon, lat_lon[0]))
    for lat, lon in closed:
        point = ET.SubElement(segment, f"{{{GPX_NS}}}trkpt", {"lat": f"{lat:.9f}", "lon": f"{lon:.9f}"})
        ET.SubElement(point, f"{{{GPX_NS}}}ele").text = "0.0"
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)


def write_preview(
    path: str | os.PathLike[str],
    outer: np.ndarray,
    inner: np.ndarray,
    centerline: np.ndarray,
    metrics: dict[str, float | int | bool],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))
    for points, label, color in (
        (outer, "Raw outer boundary", "#94a3b8"),
        (inner, "Raw inner boundary", "#64748b"),
        (centerline, "Smoothed centerline", "#e11d48"),
    ):
        closed = np.vstack((points, points[0]))
        ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2.4 if label.startswith("Smoothed") else 1.5, label=label)
    ax.scatter(centerline[0, 0], centerline[0, 1], s=65, color="#16a34a", edgecolor="white", linewidth=1.5, zorder=5, label="GPX start")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Autodrome Chaudiere preprocessing preview")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best")
    summary = (
        f"centerline: {float(metrics['centerline_length_m']):.1f} m\n"
        f"points: {int(metrics['point_count'])}\n"
        f"mean spacing: {float(metrics['mean_spacing_m']):.2f} m"
    )
    ax.text(0.02, 0.02, summary, transform=ax.transAxes, va="bottom", ha="left", bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9})
    fig.tight_layout()
    preview = Path(path)
    preview.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview, dpi=170)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    named_tracks = read_boundary_tracks(args.input)
    projected, lat0, lon0 = project_local([points for _, points in named_tracks[:2]])
    centerline, metrics = build_centerline(
        projected[0],
        projected[1],
        alignment_samples=args.alignment_samples,
        smooth_window_m=args.smooth_window_m,
        spacing_m=args.spacing_m,
    )
    lat_lon = unproject_local(centerline, lat0, lon0)
    write_gpx(args.output, lat_lon, "Autodrome Chaudiere - smoothed centerline")
    if args.preview:
        write_preview(args.preview, projected[0], projected[1], centerline, metrics)
    print(f"Input boundaries: {named_tracks[0][0]} ({len(named_tracks[0][1])} points), {named_tracks[1][0]} ({len(named_tracks[1][1])} points)")
    print(f"Outer boundary length: {float(metrics['outer_length_m']):.1f} m")
    print(f"Inner boundary length: {float(metrics['inner_length_m']):.1f} m")
    print(f"Smoothed centerline: {float(metrics['centerline_length_m']):.1f} m")
    print(f"Output: {int(metrics['point_count'])} points at {float(metrics['mean_spacing_m']):.2f} m mean spacing")
    print(f"Wrote GPX: {args.output}")
    if args.preview:
        print(f"Wrote preview: {args.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
