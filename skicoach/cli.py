import argparse
from pathlib import Path

from .pipeline import analyze_video


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="skicoach")
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze", help="Analyze a skiing video")
    analyze.add_argument("--input", required=False, help="Path to input video")
    analyze.add_argument("--video", required=False, help="Legacy alias for input video path")
    analyze.add_argument("--pose", default="mediapipe", help="Pose backend")
    analyze.add_argument(
        "--output",
        default=str(Path("data") / "outputs"),
        help="Output root directory",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "analyze":
        video_path = args.input or args.video
        if not video_path:
            raise ValueError("Provide --input <video> (or --video for compatibility)")
        result = analyze_video(video_path, args.output)
        print(f"Report JSON: {result['report_json']}")
        print(f"Report MD: {result['report_md']}")
        print(f"Annotated Video: {result['annotated_video']}")


if __name__ == "__main__":
    main()
