import numpy as np
import cv2
import csv
from icecream import ic
import sys


N = 100


def get_data(file_path):
    f = csv.reader(open(file_path, "r"))
    cities = np.zeros((N, 2))
    for i, row in enumerate(f):
        for j, el in enumerate(row):
            cities[i, j] = float(el)
    return cities


def vis(data,
        cons=np.array([]),
        dx=3,
        scale=2,
        radius=2,
        thickness=-1):
    x_max, y_max = np.max(data[:, 0]), np.max(data[:, 1])
    _size = [int(x_max + dx), int(y_max + dx), 3]
    canvas = np.zeros(tuple(_size))

    for idx, one in enumerate(data):
        shift = [0, _size[1]]
        start_point = tuple([int(shift[0] + one[0]), int(shift[1] - one[1])])
        color = (255, 255, 255)
        canvas = cv2.circle(canvas, start_point, radius, color, thickness)

    for idx, con in enumerate(cons):
        start_point = con[:2]
        end_point = con[2:]
        color = (idx % 3 * 255 / 6, (idx + 1) % 3 * 255 / 6, (idx + 2) % 3 * 255 / 6)
        canvas = cv2.arroweLine(canvas, start_point, end_point, color, -1)

    _size = np.array([i * scale for i in _size])[:2]
    canvas = cv2.resize(canvas, tuple(_size))
    cv2.imshow(".", canvas)
    cv2.waitKey(0)


if __name__ == "__main__":
    cities = get_data(sys.argv[1])
    vis(cities)