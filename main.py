import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

import numpy as np
from matplotlib.image import imread, imsave
from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from PIL import Image

from scipy.signal import savgol_filter

import os


def leave_only_blue(img):
    length = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j, 2] < 0.3:
                img[i, j, 1] = 0
                img[i, j, 0] = 0
                length += 1
            else:
                img[i, j, 1] = 1
                img[i, j, 0] = 1
            img[i, j, 2] = 1
    return length


def get_dataset(img, l):
    x = 0
    h = img.shape[0]
    result = np.zeros((l, 2))
    for i in range(h):
        for j in range(img.shape[1]):
            if img[i, j, 0] == 0:
                result[x] = [j, h - i]
                x += 1
    return result


def make_img_gray(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            color = (img[i, j, 0] + img[i, j, 1] + img[i, j, 2]) / 3
            #color = (0.299 * img[i, j, 0] + 0.587 * img[i, j, 1] + 0.114 * img[i, j, 2])
            for c in range(3):
                img[i, j, c] = color
    return img


def get_dataset_gray(img):
    length = 0
    h = img.shape[0]
    w = img.shape[1]
    x = 0
    list = []
    for i in range(h):
        for j in range(w):
            if img[i, j, 2] < 0.54:            #0.18
                length += 1
                list.append([j, h - i])
                x += 1
    result = np.asarray(list)
    return result


def make_array_with_clusters(num_cl, labels_for_points):
    result = np.zeros((num_cl, 4))
    for j in range(num_cl):
        result[j, 0] = 2000000

    for k in range(labels_for_points.shape[0]):
        lb = labels_for_points[k]
        if lb == -1:
            continue
        if result[lb, 0] > image[k, 0]:
            result[lb, 0] = image[k, 0]
        if result[lb, 1] < image[k, 0]:
            result[lb, 1] = image[k, 0]

    for x in range(num_cl):
        #ширина кластера
        result[x, 2] = result[x, 1] - result[x, 0]
        #середина кластера
        result[x, 3] = result[x, 0] + result[x, 2] / 2
    return result


def combine_clusters(arr, num_cl):
    max_value = np.max(arr[:, 2])
    result = []
    for i in range(num_cl):
        f = True
        for j in result:
            if j[3] - max_value / 2 < arr[i][3] < j[3] + max_value / 2:
                f = False
                # v = res.index(j)
                j[0] = min(arr[i][0], j[0])
                j[1] = max(arr[i][1], j[1])
                j[2] = j[1] - j[0]
                j[3] = j[0] + j[2] / 2
        if f:
            result.append(arr[i])
    result.sort(key=lambda x: x[0])
    return result


def cut_img_on_column(arr, img_path):
    pmax_r = 0
    for j in arr:
        if j[2] > pmax_r:
            pmax_r = j[2]
    pmax_r /= 2
    print(pmax_r)

    res = []
    for j in arr:
        #res.append([j[3] - pmax_r, j[3] + pmax_r])
        res.append([j[0], j[1]])
    print(res)
    print('\n')

    #print(res[0][0])
    #print(int(res[0][0]))

    img = Image.open(img_path + '.png')
    w, h = img.size
    for j in range(len(res)):
        img_res = img.crop((int(res[j][0]), 0, int(res[j][1]), h))
        img_res.save(img_path + '/' + str(j) + '.png')
    return res


def make_diagam(i, img_path):
    column_img = imread(img_path + '/' + str(i) + '.png')
    plt.imshow(column_img)
    plt.show()

    height = column_img.shape[0]
    width = column_img.shape[1]
    result = np.zeros(height)

    for h in range(height):
        for w in range(width):
            result[h] = width - (column_img[h, w, 0] + column_img[h, w, 1] + column_img[h, w, 2])
    #plt.plot(range(height), result)
    #plt.savefig(str(i) + '_d.png')
    #plt.show()

    yhat = savgol_filter(result, 30, 3)  # window size 50, polynomial order 3

    plt.plot(range(height), result)
    plt.plot(range(height), yhat, color='red')
    plt.savefig(img_path + '/' + str(i) + '_d.png')
    plt.show()

    return result


def make_centers(width, high, num_w):
    kw = width / num_w
    bw = kw / 2

    num_h = 6
    kh = high / num_h
    bh = kh / 2

    list = []
    for w in range(num_w):
        for h in range(num_h):
            list.append([kw*w + bw, kh*h + bh])
    return list


if __name__ == '__main__':
    img_name = '441'
    num_column = 6

    if not os.path.exists(img_name):
        os.mkdir(img_name)

    orig_img = imread(img_name + '.png')
    plt.imshow(orig_img)
    plt.show()

    #img_copy = orig_img.copy()
    #l = leave_only_blue(img_copy)
    #plt.imshow(img_copy)
    #plt.show()

    img_copy = orig_img.copy()
    make_img_gray(img_copy)
    plt.imshow(img_copy)
    plt.show()


    #image = get_dataset(img_copy, l)
    image = get_dataset_gray(img_copy)
    plt.scatter(image[:, 0], image[:, 1], label='True Position', s=0.05)
    plt.show()

    centers = make_centers(img_copy.shape[1], img_copy.shape[0], num_column)

    X = StandardScaler().fit_transform(image)

    db = DBSCAN(eps=0.07, min_samples=100).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    #num_cl = 7
    #kmeans = KMeans(n_clusters=num_cl)
    #kmeans.fit(image)

    plt.scatter(image[:, 0], image[:, 1], c=db.labels_, cmap='rainbow', s=0.05)
    plt.show()

    arr = make_array_with_clusters(n_clusters_, db.labels_)
    print(arr)
    print('\n')

    temp = combine_clusters(arr, n_clusters_)
    print(len(temp))
    print(temp)

    res = cut_img_on_column(temp, img_name)

    #part 2
    print('part 2\n')
    for i in range(len(res)):
        make_diagam(i, img_name)
