import multiposenet
import cv2
import numpy as np
import argparse
'''
0 	nose
1 	leftEye
2 	rightEye
3 	leftEar
4 	rightEar
5 	leftShoulder
6 	rightShoulder
7 	leftElbow
8 	rightElbow
9 	leftWrist
10 	rightWrist
11 	leftHip
12 	rightHip
13 	leftKnee
14 	rightKnee
15 	leftAnkle
16 	rightAnkle
---
See connections in multi_estimator
'''


def process(args, image_path):
    model = args.model_file
    image = cv2.imread(image_path.name)
    pose_estimator = multiposenet.PoseEstimator(model)
    poses_coords,poses_scores, elapsed = pose_estimator(cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB))
    # poses_coords = [ [person]_1,[person]_2 ,..., [person]_n], Each [person] consist of (17,2) array containing y,x for each keypoint numbered above.
    # poses_scores = [ [person]_1,[person]_2 ,..., [person]_n], Each [person] consist of (17,1) array containing score for each keypoint numbered above.

    c = np.random.randint(256,size=(len(poses_coords),3))
    for i,person in enumerate(poses_coords):
        for f_list in np.array(person):
            coord = f_list
            p = (coord[1],coord[0])
            image = cv2.circle(image,p,radius = 0,color =c[i].tolist(),thickness= 10)
        for connection in multiposenet.parentChildrenTuples:
            y0, x0 = person[connection[0]]
            y1, x1 = person[connection[1]]
            image = cv2.line(image, (int(x0), int(y0)), (int(x1), int(y1)), color = c[i].tolist(), thickness=1)
    cv2.imwrite("ntest2.jpg",image)
    return elapsed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=argparse.FileType("rb"))
    parser.add_argument("-b", "--bench", nargs="?", type=int, help="benchmark n times", default=1)
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
                f"{image.name} average time (out of {args.bench}): {np.mean(times):.2f}ms \nstd: {np.std(times):.2f}ms"
            )
        else:
            print(f"{image.name}: {times[0]:.2f}ms")

if __name__ == "__main__":
    main()
