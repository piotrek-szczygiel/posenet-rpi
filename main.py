import argparse
import sys

import numpy as np
from PIL import Image, ImageDraw

from posenet import BodyPart, PoseNet, JOINTS

MIN_CONFIDENCE = 0.40


def process(args, image_path):
    posenet = PoseNet(
        model_path=args.model_file,
        image_path=image_path,
    )
    person, elapsed = posenet.estimate_pose(verbose=args.verbose)

    if args.quiet:
        return elapsed

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for line in JOINTS:
        if (
            person.key_points[line[0].value[0]].score > MIN_CONFIDENCE
            and person.key_points[line[1].value[0]].score > MIN_CONFIDENCE
        ):
            start_point_x, start_point_y = (
                int(person.key_points[line[0].value[0]].position.x),
                int(person.key_points[line[0].value[0]].position.y),
            )
            end_point_x, end_point_y = (
                int(person.key_points[line[1].value[0]].position.x),
                int(person.key_points[line[1].value[0]].position.y),
            )
            draw.line(
                (start_point_x, start_point_y, end_point_x, end_point_y),
                fill=(255, 255, 0),
                width=3,
            )

    for key_point in person.key_points:
        if key_point.score > MIN_CONFIDENCE:
            left_top_x, left_top_y = (
                int(key_point.position.x) - 5,
                int(key_point.position.y) - 5,
            )
            right_bottom_x, right_bottom_y = (
                int(key_point.position.x) + 5,
                int(key_point.position.y) + 5,
            )
            draw.ellipse(
                (left_top_x, left_top_y, right_bottom_x, right_bottom_y),
                fill=(0, 128, 0),
                outline=(255, 255, 0),
            )

    image.show()
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=argparse.FileType("rb"))
    parser.add_argument("-b", "--bench", nargs="?", type=int, help="benchmark n times", default=1)
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-q", "--quiet", action="store_true", help="don't show output")
    parser.add_argument(
      '-m',
      '--model_file',
      default='./posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite',
      help='.tflite model to be executed')
    args = parser.parse_args()

    for image in args.images:
        times = []
        for i in range(args.bench):
            times.append(process(args, image))

        if args.bench > 1:
            print(
                f"{image.name} average time (out of {args.bench}): {np.mean(times):.2f}ms"
            )
        else:
            print(f"{image.name}: {times[0]:.2f}ms")


if __name__ == "__main__":
    main()
