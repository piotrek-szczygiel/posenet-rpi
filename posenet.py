import math
import time
from enum import Enum

import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image


class BodyPart(Enum):
    NOSE = (0,)
    LEFT_EYE = (1,)
    RIGHT_EYE = (2,)
    LEFT_EAR = (3,)
    RIGHT_EAR = (4,)
    LEFT_SHOULDER = (5,)
    RIGHT_SHOULDER = (6,)
    LEFT_ELBOW = (7,)
    RIGHT_ELBOW = (8,)
    LEFT_WRIST = (9,)
    RIGHT_WRIST = (10,)
    LEFT_HIP = (11,)
    RIGHT_HIP = (12,)
    LEFT_KNEE = (13,)
    RIGHT_KNEE = (14,)
    LEFT_ANKLE = (15,)
    RIGHT_ANKLE = (16,)


JOINTS = [
    [BodyPart.LEFT_WRIST, BodyPart.LEFT_ELBOW],
    [BodyPart.LEFT_ELBOW, BodyPart.LEFT_SHOULDER],
    [BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER],
    [BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW],
    [BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST],
    [BodyPart.LEFT_SHOULDER, BodyPart.LEFT_HIP],
    [BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP],
    [BodyPart.RIGHT_HIP, BodyPart.RIGHT_SHOULDER],
    [BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE],
    [BodyPart.LEFT_KNEE, BodyPart.LEFT_ANKLE],
    [BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE],
    [BodyPart.RIGHT_KNEE, BodyPart.RIGHT_ANKLE],
]


class Position:
    def __init__(self):
        self.x = 0
        self.y = 0


class KeyPoint:
    def __init__(self):
        self.body_part = BodyPart.NOSE
        self.position = Position()
        self.score = 0.0


class Person:
    def __init__(self):
        self.key_points = []
        self.score = 0.0


class PoseNet:
    def __init__(self, model_path, image_path):
        self.input_mean = 127.5
        self.input_std = 127.5
        self.image_path = image_path
        self.image_width = 0
        self.image_height = 0
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-x))

    def load_input_image(self):
        height, width = (
            self.input_details[0]["shape"][1],
            self.input_details[0]["shape"][2],
        )
        input_image = Image.open(self.image_path)
        self.image_width, self.image_height = input_image.size
        resize_image = input_image.resize((width, height))
        return np.expand_dims(resize_image, axis=0)

    def estimate_pose(self, verbose=False):
        input_data = self.load_input_image()

        if self.input_details[0]["dtype"] == type(np.float32(1.0)):
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)

        start_time = time.perf_counter()
        self.interpreter.invoke()
        elapsed = (time.perf_counter() - start_time) * 1000

        heat_maps = self.interpreter.get_tensor(self.output_details[0]["index"])
        offset_maps = self.interpreter.get_tensor(self.output_details[1]["index"])

        height = len(heat_maps[0])
        width = len(heat_maps[0][0])
        num_key_points = len(heat_maps[0][0][0])

        key_point_positions = [[0] * 2 for i in range(num_key_points)]
        for key_point in range(num_key_points):
            max_val = heat_maps[0][0][0][key_point]
            max_row = 0
            max_col = 0
            for row in range(height):
                for col in range(width):
                    heat_maps[0][row][col][key_point] = self.sigmoid(
                        heat_maps[0][row][col][key_point]
                    )
                    if heat_maps[0][row][col][key_point] > max_val:
                        max_val = heat_maps[0][row][col][key_point]
                        max_row = row
                        max_col = col
            key_point_positions[key_point] = [max_row, max_col]

        x_coords = [0] * num_key_points
        y_coords = [0] * num_key_points
        confidence_scores = [0] * num_key_points
        for i, position in enumerate(key_point_positions):
            position_y = int(key_point_positions[i][0])
            position_x = int(key_point_positions[i][1])
            y_coords[i] = (
                position[0] / float(height - 1) * self.image_height
                + offset_maps[0][position_y][position_x][i]
            )
            x_coords[i] = (
                position[1] / float(width - 1) * self.image_width
                + offset_maps[0][position_y][position_x][i + num_key_points]
            )
            confidence_scores[i] = heat_maps[0][position_y][position_x][i]
            if verbose:
                print(f"confidence_scores[{i}]\t= ", confidence_scores[i])

        person = Person()
        key_point_list = []
        for i in range(num_key_points):
            key_point = KeyPoint()
            key_point_list.append(key_point)
        total_score = 0
        for i, body_part in enumerate(BodyPart):
            key_point_list[i].body_part = body_part
            key_point_list[i].position.x = x_coords[i]
            key_point_list[i].position.y = y_coords[i]
            key_point_list[i].score = confidence_scores[i]
            total_score += confidence_scores[i]

        person.key_points = key_point_list
        person.score = total_score / num_key_points

        return person, elapsed
