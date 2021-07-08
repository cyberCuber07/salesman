import numpy as np
import cv2
import csv
from icecream import ic
import sys
from parms import N



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
        scale=0.6,
        radius=2,
        thickness=-1,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        fontThickness=1):
    x_max, y_max = np.max(data[:, 0]), np.max(data[:, 1])
    _size = [int(x_max + dx), int(y_max + dx), 3]
    canvas = np.zeros(tuple(_size))
    shift = [0, _size[1]]

    for idx, one in enumerate(data):
        start_point = tuple([int(shift[0] + one[0]), int(shift[1] - one[1])])
        color = (255, 255, 255)
        canvas = cv2.circle(canvas, start_point, radius, color, thickness)

    cons = cons.astype('int')
    cons = cons.reshape((cons.shape[0], 4))
    for idx, con in enumerate(cons):
        for i in range(2):
            tmp_idx = 2 * i + 1
            con[tmp_idx] = int(shift[1] - con[tmp_idx])
        start_point = tuple(con[:2])
        end_point = tuple(con[2:])
        color = (idx % 3 * 255 / 6, (idx + 1) % 3 * 255 / 6, (idx + 2) % 3 * 255 / 6)
        canvas = cv2.arrowedLine(canvas, start_point, end_point, color, 1)
        text = str(idx)
        center = tuple([int((i + j) / 2) for (i, j) in zip(start_point, end_point)])
        canvas = cv2.putText(canvas, text, center, font, fontScale, color, fontThickness, cv2.LINE_AA)

    _size = np.array([int(i * scale) for i in _size])[:2]
    canvas = cv2.resize(canvas, tuple(_size))
    cv2.imshow(".", canvas)
    cv2.waitKey(0)


if __name__ == "__main__":
    cities = get_data(sys.argv[1])
    vis(cities)