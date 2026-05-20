#!/usr/bin/env python3
"""
Convert ordered PNG frames into an animated GIF or MP4.

Examples:
  python3 png_to_movie.py --input-dir mhd_256_density --format gif --fps 12
  python3 png_to_movie.py --input-dir mhd_256_density --format mp4 --fps 24
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn PNG frames into a GIF or MP4 movie."
    )
    parser.add_argument(
        "--input-dir",
        default="mhd_256_density",
        help="Directory containing PNG frames (default: mhd_256_density).",
    )
    parser.add_argument(
        "--format",
        choices=("gif", "mp4"),
        default="gif",
        help="Output format (default: gif).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output filename. If omitted, uses <input-dir>.<format>.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=12.0,
        help="Frames per second (default: 12).",
    )
    return parser.parse_args()


def collect_frames(input_dir: Path) -> list[Path]:
    frames = sorted(input_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No .png files found in {input_dir}")
    return frames


def save_gif(frames: list[Path], output_path: Path, fps: float) -> None:
    duration_ms = int(round(1000.0 / fps))
    images = [Image.open(frame).convert("RGB") for frame in frames]
    first, rest = images[0], images[1:]
    first.save(
        output_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
    )
    for image in images:
        image.close()


def save_mp4(frames: list[Path], output_path: Path, fps: float) -> None:
    try:
        import imageio.v2 as imageio  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "MP4 output requires imageio + imageio-ffmpeg. Install with:\n"
            "  python3 -m pip install imageio imageio-ffmpeg"
        ) from exc

    with imageio.get_writer(output_path, fps=fps, codec="libx264") as writer:
        for frame in frames:
            writer.append_data(imageio.imread(frame))


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    frames = collect_frames(input_dir)
    output_name = args.output or f"{input_dir.name}.{args.format}"
    output_path = Path(output_name)

    if args.format == "gif":
        save_gif(frames, output_path, args.fps)
    else:
        save_mp4(frames, output_path, args.fps)

    print(f"Wrote {output_path} from {len(frames)} frame(s).")


if __name__ == "__main__":
    main()
