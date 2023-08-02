import cv2
import argparse
import os
import numpy as np
import math
import torch
import torchvision
from torchvision import datasets, transforms
from torchsummary import summary
import pytesseract
from pytesseract import Output
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

parser = argparse.ArgumentParser(description='Process image')
parser.add_argument('--input', metavar='FILE', type=str, nargs=1, required=True, help='input image')

args = parser.parse_args()

gray_range_r = [50, 210]
gray_range_g = [60, 230]
gray_range_b = [70, 230]

def in_gray_range(rgb_val):
    if rgb_val[0] >= gray_range_r[0] and rgb_val[0] <= gray_range_r[1]:
        if rgb_val[1] >= gray_range_g[0] and rgb_val[1] <= gray_range_g[1]:
            if rgb_val[2] >= gray_range_b[0] and rgb_val[2] <= gray_range_r[1]:
                return True

    return False


def is_white(rgb_val):
    if rgb_val[0] == 255 and rgb_val[1] == 255 and rgb_val[2] == 255:
        return True

    return False

def is_black(rgb_val):
    if rgb_val[0] == 0 and rgb_val[1] == 0 and rgb_val[2] == 0:
        return True
    
    return False

# extract point locations from an image
def extract_points_test1():
    img = cv2.imread(args.input[0])
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('grayscale chart', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print(img)

    # print(points)
    points = []

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if not in_gray_range(img[i][j]) and not is_white(img[i][j]) and not is_black(img[i][j]):
                points.append([i, j])
                img[i][j] = 0
            else:
                img[i][j] = 255

    cv2.imshow('grayscale chart', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(points)

def extract_points_test2():
    img = cv2.imread(args.input[0])
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(img_gray, 0, 100)
    # print("POINT 1: ")
    # print(edges[42:52, 35:50])
    # print("POINT 2: ")
    # print(edges[52:70, 45:60])
    # print(edges[610:635, 1005:1030])

    # split into boxes with side length r (if radius = 4 then each side = 4)
    # identify as point if there are at least 'r/2' number of 255's on each side of the box
    min_radius = 5
    max_radius = 10

    points = []
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            for r in range(min_radius, max_radius):
                if i + r >= edges.shape[0] or j + r >= edges.shape[1]:
                    continue

                # current i,j is top left corner of the box
                curr_box = edges[i:i+r, j:j+r]
                
                # check for r/2 255's on each side of the box
                top = curr_box[0, 0:r-1].sum()
                left = curr_box[0:r-1, 0].sum()
                bottom = curr_box[r-1, 0:r-1].sum()
                right = curr_box[0:r-1, r-1].sum()

                threshold = 255*r/2

                if top >= threshold and left >= threshold and bottom >= threshold and right >= threshold:
                    points.append([i, j, r])

    return points

def extract_points_test3():
    img = cv2.imread(args.input[0])
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(img_gray, 0, 100)

    # split into boxes with side length r (if radius = 4 then each side = 4)
    # identify as point if there are at least 'r/2' number of 255's on each side of the box
    min_radius = 5
    max_radius = 10

    accumulator = np.zeros((edges.shape[0], edges.shape[1], max_radius-min_radius))
    print(accumulator.shape)

    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            for r in range(min_radius, max_radius):
                for theta in range(0, 360):
                    a = int(i - r * math.cos(theta))
                    b = int(j - r * math.sin(theta))

                    if a >= 0 and a < edges.shape[0] and b >= 0 and b < edges.shape[1]:
                        # print(a, b, r)
                        accumulator[a][b][r-min_radius] += 1

    print(max(accumulator))

def extract_points_test4():
    img = cv2.imread(args.input[0])
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(img_gray, 0, 100)

    # split into boxes with side length r (if radius = 4 then each side = 4)
    # identify as point if there are at least 'r/2' number of 255's on each side of the box
    min_radius = 5
    max_radius = 10

    points = []

    # circle detection
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            for r in range(min_radius, max_radius):
                if i + r >= edges.shape[0] or j + r >= edges.shape[1]:
                    continue

                # current i,j is top left corner of the box
                curr_box = edges[i:i+r, j:j+r]

                curr_radius = r//2

                # top left corner of middle box
                box_radius = curr_radius//2
                middle = curr_radius
                middle_box = curr_box[middle-box_radius:middle+box_radius, middle-box_radius:middle+box_radius]
                if np.sum(middle_box) > 0:
                    continue

                top = curr_box[0, 0:r-1].sum()
                left = curr_box[0:r-1, 0].sum()
                bottom = curr_box[r-1, 0:r-1].sum()
                right = curr_box[0:r-1, r-1].sum()

                threshold = 255*4
                if top >= threshold and left >= threshold and bottom >= threshold and right >= threshold:
                    points.append([i, j, r])

    return points

# manual convolution
def convolve(img, kernel):
    output = np.zeros(img.shape)
    padded_img = np.zeros((img.shape[0]+2, img.shape[1]+2))
    padded_img[1:img.shape[0]+1, 1:img.shape[1]+1] = img
    for i in range(0, padded_img.shape[0]-kernel.shape[0]+1):
        for j in range(0, padded_img.shape[1]-kernel.shape[1]+1):
            end_i = i+kernel.shape[0]
            end_j = j+kernel.shape[1]
            output[i][j] = (padded_img[i:end_i, j:end_j] * kernel).sum()

    return output
    
# Canny edge detection
def edge_detection():
    img = cv2.imread(args.input[0])
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    smoothing_kernel = np.array([[2, 4, 5, 4, 2], [4, 9, 12, 9, 4], [5, 12, 15, 12, 5], [4, 9, 12, 9, 4], [2, 4, 5, 4, 2]])
    smoothing_kernel = smoothing_kernel*1/159
    edge_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    img = convolve(img_gray, smoothing_kernel)
    img = convolve(img_gray, edge_kernel)

    cv2.imshow('edges', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# extract locations of points using edge detection
def extract_points():
    img = cv2.imread(args.input[0])
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(img_gray, 0, 100)
    # print("POINT 1: ")
    # print(edges[42:52, 35:50])
    # print("POINT 2: ")
    # print(edges[52:70, 45:60])
    # print(edges[610:635, 1005:1030])

    """
    filtering to make point detection more accurate
    """

    # detect and remove lines from chart
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=10, minLineLength=10)
    print(lines)

    for i in range(len(lines)):
        pt1 = (lines[i][0][0], lines[i][0][1])
        pt2 = (lines[i][0][2], lines[i][0][3])

        cv2.line(edges, pt1, pt2, 0, 3)

    # TODO: detect and remove text (axis labels) from chart

    points = []

    # Using hough circle transform
    points = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, minDist=1, param1=100, param2=12, minRadius=3, maxRadius=10)

    # points are in (x, y) coordinates, i.e. (column, row) indices
    points = np.int64(np.around(points[0]))
    # print(points)

    for p in points:
        cv2.circle(edges, (p[0], p[1]), p[2], (255, 255, 255), 2)

    # convert into (row, column) format
    points = [[p[1], p[0]] for p in points]

    cv2.imshow('edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # print(edges[40:65, 30:50])
    print(edges[40:55, 435:450])
    # print(edges[:, 290:300])
    print(points)

    return points

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Identify clusters based on distance between points
# If distance is under distance_threshold, then points are in the same cluster
# If points_threshold number of points are in the same cluster then it is a 
# significant enough cluster
def find_clusters(points, distance_threshold, points_threshold):
    clusters = [[p] for p in points]
    
    for cluster in clusters:
        for p1 in points:
            for p2 in cluster:
                if (p1[0] != p2[0] or p1[1] != p2[1]) and dist(p1, p2) <= distance_threshold:
                    cluster.append(p1)

    final_clusters = []
    for cluster in clusters:
        if len(cluster) >= points_threshold:
            final_clusters.append(cluster)

    return final_clusters


points = extract_points()
# clusters = find_clusters(points, 20, 3)
# for cluster in clusters:
#     print(cluster)
