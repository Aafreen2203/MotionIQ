"""Command line utilities for reproducible local preprocessing."""

from __future__ import annotations

import argparse
import json

from motioniq.video import build_manifest, extract_frames, inspect_video


def main() -> None:
    parser = argparse.ArgumentParser(prog="motioniq")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("video")
    extract = commands.add_parser("extract-frames")
    extract.add_argument("video")
    extract.add_argument("output")
    extract.add_argument("--every", type=int, default=1)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("root")
    manifest.add_argument("output")
    args = parser.parse_args()
    if args.command == "inspect":
        print(json.dumps(inspect_video(args.video), indent=2))
    elif args.command == "extract-frames":
        print(
            json.dumps(
                [str(path) for path in extract_frames(args.video, args.output, args.every)],
                indent=2,
            )
        )
    else:
        print(json.dumps(build_manifest(args.root, args.output), indent=2))


if __name__ == "__main__":
    main()
