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
from text_extraction import extract_text_cnn
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

parser = argparse.ArgumentParser(description='Process image')
parser.add_argument('--input', metavar='FILE', type=str, nargs=1, required=True, help='input image')
parser.add_argument('--model_type', type=str, nargs=1, required=True, help='model type for text extraction')

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

    edges = cv2.Canny(img_gray, 200, 200)

    cv2.imshow('edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    """
    Filtering to make point detection more accurate
    1. Detect and remove lines from chart
    2. TODO: Detect and remove text (axis labels, title, etc.) from chart
        a. Identify bottom-most line and crop out everything below it
        b. Identify top-most line and crop out everything above it
        c. Identify left-most line and crop out everything to the left of it
    """
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=10, minLineLength=10)

    # Contains row index of bottom-most (highest row index) and topmost (lowest row index) lines
    bottom = 0
    top = len(edges)-1

    # Contains column index of leftmost (lowest column index) line
    left = len(edges[0])-1

    for i in range(len(lines)):
        pt1 = (lines[i][0][0], lines[i][0][1])
        pt2 = (lines[i][0][2], lines[i][0][3])

        cv2.line(edges, pt1, pt2, 0, 3)

        if pt1[1] > bottom and pt1[1] == pt2[1]:
            bottom = pt1[1]

        if pt1[1] < top and pt1[1] == pt2[1]:
            top = pt1[1]

        if pt1[0] < left and pt1[0] == pt2[0]:
            left = pt1[0]

    cv2.imshow('uncropped', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    edges = edges[:bottom, left:]
    cv2.imshow('cropped', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Detect circular points using hough circle transform
    # Points are in (x, y) coordinate, i.e. (column, row) indices
    points = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, minDist=1, param1=100, param2=12, minRadius=3, maxRadius=10)
    points = np.int64(np.around(points[0]))

    for p in points:
        cv2.circle(edges, (p[0], p[1]), p[2], (255, 255, 255), 2)

    # Convert into (row, column) format
    points = [(p[1], p[0]) for p in points]

    # Filter out duplicate points
    points = set(points)

    cv2.imshow('points', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(points)

    return list(points)

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# return line separating y axis from the rest of the chart (leftmost line)
# lines are result of HoughLinesP (endpoints are in (x,y) coordinates)
def get_y_axis(img, lines):
    # Contains column index of leftmost line (lowest column index)
    left = len(img[0])-1

    for i in range(len(lines)):
        pt1 = (lines[i][0][0], lines[i][0][1])
        pt2 = (lines[i][0][2], lines[i][0][3])

        if pt1[0] < left and pt1[0] == pt2[0]:
            left = pt1[0]

    return left

# return line separating x axis from the rest of the chart (bottom-most line)
def get_x_axis(img, lines):
    # Contains row index of bottom-most line (highest row index)
    bottom = 0

    for i in range(len(lines)):
        pt1 = (lines[i][0][0], lines[i][0][1])
        pt2 = (lines[i][0][2], lines[i][0][3])

        if pt1[1] > bottom and pt1[1] == pt2[1]:
            bottom = pt1[1]

    return bottom

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

# img: input image
# cluster: list of points in cluster in (row, column) format (not (x,y) format)
# left: column index of leftmost line (used to extract y axis labels)
# bottom: row index of bottom-most line (used to extract x axis labels)
'''
1. Identify upper bound and lower bound of y axis label (in terms of row index)
  a. Identify bounds of cluster
  b. Run OCR every 50 pixels above and below cluster bounds
2. Identify upper bound and lower bound of x axis label (in terms of column index)
'''
def identify_cluster_location(img, cluster):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(img_gray, 200, 200)

    # axes = cv2.Canny(img_gray, 500, 500)
    # axes_lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=5, minLineLength=10)
    # cv2.imshow('axes', axes)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # for i in range(len(axes_lines)):
    #     pt1 = (axes_lines[i][0][0], axes_lines[i][0][1])
    #     pt2 = (axes_lines[i][0][2], axes_lines[i][0][3])

    #     cv2.line(axes, pt1, pt2, 0, 3)

    # cv2.imshow('without axes', axes)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=10, minLineLength=10)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=10, minLineLength=10)
    # print(lines)
    left = get_y_axis(edges, lines)
    bottom = get_x_axis(edges, lines)
    print(left, bottom)

    upper_y_cluster = len(edges) - 1 # upper y bound of cluster (smallest row index in cluster)
    lower_y_cluster = 0 # lower y bound of cluster (largest row index in cluster)

    lower_x_cluster = len(edges[0]) - 1 # lower x bound of cluster (smallest column index in cluster)
    upper_x_cluster = 0 # upper x bound of cluster (largest column index in cluster)

    for point in cluster:
        upper_y_cluster = min(point[0], upper_y_cluster)
        lower_y_cluster = max(point[0], lower_y_cluster)

        lower_x_cluster = min(point[1], lower_x_cluster)
        upper_x_cluster = max(point[1], upper_x_cluster)
    
    print("y cluster bounds: ", upper_y_cluster, lower_y_cluster)
    print("x cluster bounds: ", lower_x_cluster, upper_x_cluster)

    cv2.imshow('cluster', edges[upper_y_cluster:lower_y_cluster, left+lower_x_cluster:left+upper_x_cluster])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    y_axis = edges[:, :left]
    x_axis = edges[bottom:, :]

    cv2.imshow('y axis', y_axis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('x axis', x_axis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    '''
    Extract y coordinates of cluster bounds:
    1 Get line right above highest bound, extract axis label from that line
        - largest row index that is lower than upper_y_cluster
    2. Get row index of line right below lowest bound, extract axis label from that line
        - smallest row index that is higher than lower_y_cluster
    '''
    upper_bound = 0
    lower_bound = len(edges)-1
    for i in range(len(lines)):
        pt1 = (lines[i][0][0], lines[i][0][1])
        pt2 = (lines[i][0][2], lines[i][0][3])

        if pt1[1] <= upper_y_cluster and pt1[1] == pt2[1] and pt1[1] > upper_bound:
            upper_bound = pt1[1]

        if pt1[1] >= lower_y_cluster and pt1[1] == pt2[1] and pt1[1] < lower_bound:
            lower_bound = pt1[1]

    print(upper_bound, lower_bound)
    upper_y_axis = img_gray[upper_bound-25:upper_bound+25, left-50:left]
    lower_y_axis = img_gray[lower_bound-25:lower_bound+25, left-50:left]

    # _, upper_y_axis = cv2.threshold(upper_y_axis, 127, 255, cv2.THRESH_BINARY_INV)

    # if args.model_type[0] == 'cnn':
    #     upper_y_axis = upper_y_axis.reshape((1, 1, 50, 50))

    # if args.model_type[0] == 'simple_model':   
    #     upper_y_axis = upper_y_axis.reshape((1, 2500))

    # log_prob = extract_text_cnn(torch.from_numpy((upper_y_axis.astype(np.float32))))
    # prob = list(torch.exp(log_prob).detach().numpy()[0])
    # upper_y_text = prob.index(max(prob))

    upper_y_text = pytesseract.image_to_string(upper_y_axis, config='--psm 7 -c tessedit_char_whitelist=0123456789')
    lower_y_text = pytesseract.image_to_string(lower_y_axis, config='--psm 7 -c tessedit_char_whitelist=0123456789')
    print('upper y: ', upper_y_text)
    print('lower y: ', lower_y_text)

    print(upper_y_axis)
    cv2.imshow('upper y axis', upper_y_axis)
    cv2.imshow('lower y axis', lower_y_axis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    '''
    Extract x coordinates of cluster bounds:
    1 Get line to the right of highest bound, extract axis label from that line
        - smallest column index that is higher than upper_x_cluster
    2. Get column index of line to the left of lowest bound, extract axis label from that line
        - largest column index that is lower than lower_x_cluster
    '''
    left_bound = 0
    right_bound = len(edges[0])-1

    for i in range(len(lines)):
        pt1 = (lines[i][0][0], lines[i][0][1])
        pt2 = (lines[i][0][2], lines[i][0][3])

        if pt1[0] <= lower_x_cluster and pt1[0] == pt2[0] and pt1[0] > left_bound:
            left_bound = pt1[0]

        if pt1[0] >= upper_x_cluster and pt1[0] == pt2[0] and pt1[0] < right_bound:
            right_bound = pt1[0]

    lower_x_axis = img_gray[bottom+10:bottom+60, left_bound-25:left_bound+25]
    # x = cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    upper_x_axis = img_gray[bottom+10:bottom+60, right_bound-25:right_bound+25]
    # print(left_bound, right_bound)

    lower_x_text = pytesseract.image_to_string(lower_x_axis, config='--psm 7 -c tessedit_char_whitelist=0123456789')
    upper_x_text = pytesseract.image_to_string(upper_x_axis, config='--psm 7 -c tessedit_char_whitelist=0123456789')
    print('lower x text: ', lower_x_text)
    print('upper x text: ', upper_x_text)
    print(upper_x_axis)
    
    # _, lower_x_axis = cv2.threshold(lower_x_axis, 127, 255, cv2.THRESH_BINARY_INV)

    # if args.model_type[0] == 'cnn':
    #     lower_x_axis = lower_x_axis.reshape((1, 1, 50, 50))

    # if args.model_type[0] == 'simple_model':   
    #     lower_x_axis = lower_x_axis.reshape((1, 2500))

    # log_prob = extract_text_cnn(torch.from_numpy((lower_x_axis.astype(np.float32))))
    # prob = list(torch.exp(log_prob).detach().numpy()[0])
    # lower_x_text = prob.index(max(prob))
    # print(lower_x_text)

    cv2.imshow('upper x axis', upper_x_axis)
    cv2.imshow('lower x axis', lower_x_axis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread(args.input[0])
points = extract_points()
# TODO: Replace with sklearn clustering code
# cluster = [(138, 40), (140, 40), (72, 48), (110, 96), (156, 72), (108, 96), (110, 16), (160, 72), (108, 16), (72, 50), (156, 74), (138, 38), (108, 98), (140, 38), (158, 74)]
cluster = [(618, 958), (602, 980), (604, 980), (618, 912), (632, 1002), (634, 1002), (602, 982), (566, 992), (604, 982), (584, 946), (632, 1004), (634, 1004), (602, 978), (620, 910), (600, 980), (582, 944), (622, 910)]
identify_cluster_location(img, cluster)

