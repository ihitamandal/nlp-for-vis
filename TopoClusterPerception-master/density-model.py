import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import TopoClusterPerception.ClusterModels as ClusterModels
import argparse
import json
import time
import cv2

# setup argument parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--histogram_res_x', metavar='X', type=int, nargs=1, required=True, help='x resolution of density plot')
parser.add_argument('--histogram_res_y', metavar='Y', type=int, nargs=1, required=True, help='y resolution of density plot')
parser.add_argument('--input', metavar='FILE', type=str, nargs=1, required=True, help='input image')
parser.add_argument('--show_plot', type=bool, default=False, help='show the threshold vs clusters plot')

args = parser.parse_args()

# Set grid resolution
res = [args.histogram_res_x[0], args.histogram_res_y[0]]

# Read Images
time_1 = time.time_ns()
img = mpimg.imread(args.input[0])
# print(img)

# Build histograms
time_2 = time.time_ns()
bins = ClusterModels.histogram(img, res)
mpimg.imsave(args.input[0] + ".hist.png", bins, cmap='Greys_r')

# Build model
time_3 = time.time_ns()
mt = ClusterModels.density_model(bins)
#print( json.dumps( mt.mt.h0, indent=2) )

# Plot the result
time_4 = time.time_ns()
threshold, clusters = mt.threshold_plot()
time_5 = time.time_ns()

if args.show_plot:
    threshold[-1] = threshold[-2] * 1.1

    plt.plot(threshold, clusters)
    plt.xlabel('threshold')
    plt.ylabel('cluster count')
    plt.suptitle('Plot of threshold vs. cluster count')
    plt.show()
else:
    threshold[-1] = 'inf'
    info = {'datafile': args.input[0],
            'histogram resolution': res,
            'time_load (ms)': (time_2 - time_1)/1000000,
            'time histogram (ms)': (time_3 - time_2)/1000000,
            'time MT (ms)': (time_4 - time_3)/1000000}

    print(json.dumps({'info': info, 'threshold': threshold, 'clusters': clusters}, indent=2))

# prints show up in json file
final_threshold = threshold[-1]
final_clusters = clusters[-1]
print("final threshold: ", threshold[-1])
print("clusters: ", clusters[-1])
print("clusters at final threshold: ", mt.get_cluster_count_at_threshold(final_threshold))

# get location of the clusters

# split chart into four sections to tell which quadrant the cluster is in
# image = cv2.imread(args.input[0])
image = cv2.imread(args.input[0])
height = image.shape[0]
width = image.shape[1]

width_half = width//2
height_half = height//2

top_left = image[:height_half, :width_half]
bottom_left = image[height_half:, :width_half]
top_right = image[:height_half, width_half:]
bottom_right = image[height_half:, width_half:]

bins_top_left = ClusterModels.histogram(top_left, res)
mt_top_left = ClusterModels.density_model(bins_top_left)
threshold_top_left, clusters_top_left = mt_top_left.threshold_plot()
plt.plot(threshold_top_left, clusters_top_left)
mpimg.imsave("fake_chart_1_top_left.hist.png", bins_top_left, cmap='Greys_r')
plt.show()
if clusters_top_left[-1] > 0:
    print("cluster in top left: ", clusters_top_left[-1], threshold_top_left[-1])

bins_bottom_left = ClusterModels.histogram(bottom_left, res)
mt_bottom_left = ClusterModels.density_model(bins_bottom_left)
threshold_bottom_left, clusters_bottom_left = mt_bottom_left.threshold_plot()
plt.plot(threshold_bottom_left, clusters_bottom_left)
mpimg.imsave("fake_chart_1_bottom_left.hist.png", bins_bottom_left, cmap='Greys_r')
plt.show()
if clusters_bottom_left[-1] > 0:
    print("cluster in bottom left: ", clusters_bottom_left[-1], threshold_bottom_left[-1])


bins_top_right = ClusterModels.histogram(top_right, res)
mt_top_right = ClusterModels.density_model(bins_top_right)
# clusters_top_right = mt_top_right.get_cluster_count_at_threshold(final_threshold)
threshold_top_right, clusters_top_right = mt_top_right.threshold_plot()
mpimg.imsave("fake_chart_1_top_right.hist.png", bins_top_right, cmap='Greys_r')
if clusters_top_right[-1] > 0:
    print("cluster in top right: ", clusters_top_right[-1], threshold_top_right[-1])

bins_bottom_right = ClusterModels.histogram(bottom_right, res)
mt_bottom_right = ClusterModels.density_model(bins_bottom_right)
# clusters_bottom_right = mt_bottom_right.get_cluster_count_at_threshold(final_threshold)
threshold_bottom_right, clusters_bottom_right = mt_bottom_right.threshold_plot()
mpimg.imsave("fake_chart_1_bottom_right.hist.png", bins_bottom_right, cmap='Greys_r')
if clusters_bottom_right[-1] > 0:
    print("cluster in bottom right: ", clusters_bottom_right[-1], threshold_bottom_right[-1])

cv2.imwrite("fake_chart_1_top_left.png", top_left)
cv2.imwrite("fake_chart_1_bottom_left.png", bottom_left)
cv2.imwrite("fake_chart_1_top_right.png", top_right)
cv2.imwrite("fake_chart_1_bottom_right.png", bottom_right)

