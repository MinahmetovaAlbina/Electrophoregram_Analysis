import matplotlib.pyplot as plt

import numpy as np
import numpy.polynomial as poly
from matplotlib.image import imread

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from PIL import Image

from scipy import signal

import os


def make_img_gray(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            color = (img[i, j, 0] + img[i, j, 1] + img[i, j, 2]) / 3
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
            if img[i, j, 2] < 0.54:
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
        if result[lb, 0] > dark_array[k, 0]:
            result[lb, 0] = dark_array[k, 0]
        if result[lb, 1] < dark_array[k, 0]:
            result[lb, 1] = dark_array[k, 0]

    for x in range(num_cl):
        # ширина кластера
        result[x, 2] = result[x, 1] - result[x, 0]
        # середина кластера
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
        res.append([j[0], j[1]])
    print(res)
    print('\n')

    img = Image.open(img_path + '.png')
    w, h = img.size
    for j in range(len(res)):
        img_res = img.crop((int(res[j][0]), 0, int(res[j][1]), h))
        img_res.save(img_path + '/' + str(j) + '.png')
    return res


def get_peaks(x, y, yhat):
    max = np.max(yhat)
    min = np.min(yhat)
    h = (max - min) * 0.2 + min
    p = (max-min) * 0.15
    result, _ = signal.find_peaks(yhat, height=h, prominence=p)
    print(result)
    print('/n')
    return result


def make_column_diagram(column_num, img, img_path, num_of_mr_ladder):
    height = img.shape[0]
    width = img.shape[1]
    y = np.zeros(height)

    for h in range(height):
        for w in range(width):
            y[h] = width - (img[h, w, 0] + img[h, w, 1] + img[h, w, 2])

    yhat = signal.savgol_filter(y, 30, 3)  # window size 30, polynomial order 3
    peaks = get_peaks(range(height), y, yhat)

    plt.plot(range(height), y)
    plt.plot(range(height), yhat, color='red')
    plt.plot(peaks, yhat[peaks], "x", color='black')

    if num_molecular_ladder == column_num:
        plt.title('Молекулярная лестница')
    elif column_num < num_of_mr_ladder:
        plt.title('колонка ' + str(column_num + 1))
    else:
        plt.title('колонка ' + str(column_num))

    plt.xlabel("Длина пробега белка (пикс)")
    plt.ylabel("Степень насыщенности ")
    plt.savefig(img_path + '/' + str(column_num) + '_d.png')
    plt.show()

    return y, yhat, peaks


def make_mr_diagram(mr_ladder, mr_peaks, img_path):
    x = mr_peaks
    c = poly.Polynomial.fit(x, mr_ladder, deg=1)
    k = c.convert().coef
    yn = x * k[1] + k[0]

    plt.plot(x, yn, color='green')
    plt.plot(x, mr_ladder, 'x', color='red')
    plt.xlabel("Длина пробега белка (пикс)")
    plt.ylabel("Молекулярная масса белка (Да)")
    plt.savefig(img_path + '/mr_d.png')
    plt.show()

    return k


def mr_on_peaks(peaks, b, c, img_path, column_num):
    mrs = peaks * b + c
    print(mrs)
    prs = []
    outp = ''
    count = 0
    for i in mrs:
        p = find_protein_on_mr(i)
        prs.append(p)
        outp += str(count) + ':' + str(round(i)) + ':' + str(p) + ','
        count += 1

    with open(str(img_path) + '/' + str(column_num) + '_result.txt', 'w') as f:
        l = len(outp)
        if l != 0:
            f.write(outp[:l-1])
    print(prs)
    return mrs


def find_protein_on_mr(mr):
    proteins = get_protein_on_Molecular_mass()
    mrs = proteins.keys()
    max = 1000000
    min = 0
    for i in mrs:
        if mr > i > min:
            min = i
        if mr < i < max:
            max = i
    if min == 0:
        return proteins[max]
    kmin = mr - min
    kmax = max - mr
    if(kmin < kmax):
        return proteins[min]
    return proteins[max]


def make_centers(width, high, num_w):
    kw = width / num_w
    bw = kw / 2

    num_h = 6
    kh = high / num_h
    bh = kh / 2

    list = []
    for w in range(num_w):
        for h in range(num_h):
            list.append([kw * w + bw, kh * h + bh])
    return list


def get_protein_on_Molecular_mass():
    proteins = {66438: 'Альбумин', 160000: 'Иммуноглобулин G', 28079: 'Аполипопротеин A-I', 79600: 'Трансферрин',
                8691: 'Аполипопротеин A-II', 40000: 'a1-Кислый Гликопротеин',
                54000: 'Транстиретин', 104000: 'Гаптоглобин', 57000: 'Гемопексин', 170000: 'Иммуноглобулин A',
                8765: 'Аполипопротеин C-III', 30000: 'a2-Макроглобулин', 50000: 'a2-HS Гликопротеин',
                51000: 'Gc глобулин', 6631: 'Аполипопротеин C-I', 340000: 'Фибриноген ', 68000: 'a1-Антихимотрипсин',
                180000: 'C3', 55000: 'Витронектин', 45000: 'Яичный альбумин', 31000: 'Карбоангидраза',
                43375: 'Аполипопротеин A-IV', 8915: 'Аполипопротеин C-I', 63000: 'b2-Гликопротеин I',
                65000: 'Антитромбин III', 81000: 'Плазминоген', 21500: 'Ингибитор трипсина',
                135000: 'Церулоплазмин', 21000: 'Ретинол-связывающий белок', 14500: 'Лактальбумин'}
    return proteins


def read_data_from_file(file_name):
    num_column = 0
    num_molecular_ladder = 0
    mrs_on_molecular_ladder = []
    with open(file_name + '.txt', 'r') as file:
        inp = file.readline()
        try:
            num_column = int(inp)
        except ValueError:
            print('колличество колонок не является числом')
        else:
            if num_column < 1:
                print('колличество колонок должно быть положительным')

        inp = file.readline()
        try:
            num_molecular_ladder = int(inp)
        except ValueError:
            print('номер колонки, являющейся молекулярой лестницей, не является числом')
        else:
            if num_molecular_ladder > num_column or num_molecular_ladder < 1:
                print('номер колонки, являющейся молекулярой лестницей неверен')

        inp = file.readline()
        temp = inp.split(',')
        for i in temp:
            n = -2
            if len(i) == 0:
                n = -1
            try:
                n = int(i)
            except ValueError:
                print('не все значения молекулярной массы на молекулярной лестнице являются числами')
            mrs_on_molecular_ladder.append(n)

    return num_column, num_molecular_ladder, mrs_on_molecular_ladder


if __name__ == '__main__':
    img_name = input("input file's name: ")

    num_column, num_molecular_ladder, mrs_on_molecular_ladder = read_data_from_file(img_name)
    num_molecular_ladder -= 1

    # создание директории для сохранения результатов
    if not os.path.exists(img_name):
        os.mkdir(img_name)

    orig_img = imread(img_name + '.png')
    plt.imshow(orig_img)
    plt.show()

    make_img_gray(orig_img)
    plt.imshow(orig_img)
    plt.show()

    dark_array = get_dataset_gray(orig_img)
    plt.scatter(dark_array[:, 0], dark_array[:, 1], label='True Position', s=0.05)
    plt.show()

    centers = make_centers(orig_img.shape[1], orig_img.shape[0], num_column)

    X = StandardScaler().fit_transform(dark_array)

    db = DBSCAN(eps=0.07, min_samples=100).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # количество кластеров, за исключением шума, если он есть
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("колличество кластеров: %d" % n_clusters_)
    print("количество выбросов: %d" % n_noise_)

    plt.scatter(dark_array[:, 0], dark_array[:, 1], c=db.labels_, cmap='rainbow', s=0.05)
    plt.show()

    arr = make_array_with_clusters(n_clusters_, db.labels_)
    print(arr)
    print('\n')

    new_clusters = combine_clusters(arr, n_clusters_)
    print(len(new_clusters))
    print(new_clusters)

    res = cut_img_on_column(new_clusters, img_name)

    # part 2
    print('part 2\n')
    columns_peaks = []
    for i in range(len(res)):
        column_img = imread(img_name + '/' + str(i) + '.png')
        plt.imshow(column_img)
        plt.show()

        yo, yc, peaks = make_column_diagram(i, column_img, img_name, num_molecular_ladder)
        columns_peaks.append(peaks)

    koef = make_mr_diagram(mrs_on_molecular_ladder, columns_peaks[num_molecular_ladder], img_name)

    for i in range(len(res)):
        if i == num_molecular_ladder:
            continue
        mr_on_peaks(columns_peaks[i], koef[1], koef[0], img_name, i)
