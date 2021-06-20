import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

mpl.use('macosx')
with open("/Users/aribakhan/Dropbox (MIT)/shared_Khan/final_data/data.json") as f:
    data = json.load(f)


class Cell:
    count = 0

    def __init__(self, image, x, y):
        self.image = image
        self.x = x
        self.y = y
        # id --> number
        self.id = str(image) + '_' + str(Cell.count)
        Cell.count += 1

    # squared Euclidean distance
    def squared_distance(self, other):
        dist = 0
        for x, y in zip((self.x, self.y), (other.x, other.y)):
            dist += (x - y) ** 2
        return dist


threshold = 30
cell_objects = []
for image in data:
    Cell.count = 0
    cells = []
    for cell in data[image]:
        cells.append(Cell(image, cell[0], cell[1]))
    cell_objects.append(cells)

cells = cell_objects[0]


# Plan
# 1) Matrix of all pairwise distances
# 2) Boolify given cutoff
# 3) Create list of connected points --> merged into larger clusts


def get_distance_matrix(points, threshold):
    n = len(points)
    t2 = threshold ** 2
    prox = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            # symmetric matrix
            prox[i][j] = (points[i].squared_distance(points[j]) < t2)
            prox[j][i] = prox[i][j]

    return prox


def find_clusters(points, threshold):
    n = len(points)
    prox = get_distance_matrix(points, threshold)
    point_in_list = [None] * n
    clusters = []
    for i in range(n):
        for j in range(i + 1, n):
            # indexes into True combination
            if prox[i][j]:
                cells1 = point_in_list[i]
                cells2 = point_in_list[j]
                # comparisons once original points are inserted
                if cells1 is not None:
                    if cells2 is None:
                        cells1.append(j)
                        point_in_list[j] = cells1
                    elif cells2 is not cells1:
                        # merge the two lists if not identical
                        cells1 += cells2
                        point_in_list[j] = cells1
                        del clusters[clusters.index(cells2)]
                    else:
                        pass  # both points are already in the same cluster
                elif cells2 is not None:
                    cells2.append(i)
                    point_in_list[i] = cells2
                else:
                    list_new = [i, j]
                    for index in [i, j]:
                        point_in_list[index] = list_new
                    clusters.append(list_new)
        if point_in_list[i] is None:
            list_new = [i]  # point is isolated so far
            point_in_list[i] = list_new
            clusters.append(list_new)
    return clusters


def generate_random_colors(N):
    np.random.seed(5423)
    df = pd.Series(np.random.randint(10, 50, N), index=np.arange(1, N + 1))

    cmap = plt.cm.tab10
    colors = cmap(np.arange(len(df)) % cmap.N)

    return colors


def graph_clusters(clusters, cells):
    plt.figure(figsize=(5.511013215859031, 5.511013215859031))
    x = [cell.x for cell in cells]
    y = [cell.y for cell in cells]
    x_cluster = [[x[index] for index in cluster] for cluster in clusters]
    y_cluster = [[y[index] for index in cluster] for cluster in clusters]

    colors = generate_random_colors(len(clusters))
    plt.scatter(x, y, s=120, color='k', alpha=0.3)
    for i, (x_c, y_c) in enumerate(zip(x_cluster, y_cluster)):
        plt.scatter(x_c, y_c, s=120, facecolors='none', edgecolors=colors[i])
    plt.show()


clusters = find_clusters(cells, threshold)
clusters_obj = [[cells[index] for index in cluster] for cluster in clusters]
graph_clusters(clusters, cells)
