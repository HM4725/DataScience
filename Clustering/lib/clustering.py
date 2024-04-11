from argparse import ArgumentParser
import math
import os
try:
    from tqdm import tqdm
except:
    def tqdm(x, desc):
        return x

NOISE = -1


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def parse():
    parser = ArgumentParser()
    parser.add_argument('file',
                        help='input data file name')
    parser.add_argument('n',
                        help='number of clusters for the corresponding input data')
    parser.add_argument('eps',
                        help='maximum radius of the neighborhood')
    parser.add_argument('minpts',
                        help='minimum number of points in an Eps-neighborhood of a given point')
    args = parser.parse_args()
    return (args.file, int(args.n), float(args.eps), float(args.minpts))


def read_csv(file, sep='\t'):
    dataset = []
    with open(file, 'r') as f:
        for line in f:
            arr = line.strip().split(sep)
            arr[0] = int(arr[0])
            arr[1] = float(arr[1])
            arr[2] = float(arr[2])
            datapoint = tuple(arr)
            dataset.append(datapoint)
    return dataset


def euclidean_distance(p1, p2):
    _, x1, y1 = p1
    _, x2, y2 = p2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def manhattan_distance(p1, p2):
    _, x1, y1 = p1
    _, x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def range_query(db, dist_fn, p, eps):
    return [q for q in db if dist_fn(p, q) <= eps]


@static_vars(cluster_id=0)
def dbscan(db, eps, minpts, dist_fn):
    max_id = db[-1][0]
    labels = [None for _ in range(max_id + 1)]
    for p in tqdm(db, desc="dbscan"):
        id = p[0]
        if labels[id] != None:
            continue
        neighbors = range_query(db, dist_fn, p, eps)
        if len(neighbors) < minpts:
            labels[id] = NOISE
        cluster_id = dbscan.cluster_id
        dbscan.cluster_id += 1
        labels[id] = cluster_id
        neighbors.remove(p)
        seed_set = neighbors
        for q in seed_set:
            id = q[0]
            if labels[id] == NOISE:
                labels[id] = cluster_id
            if labels[id] != None:
                continue
            neighbors = range_query(db, dist_fn, q, eps)
            labels[id] = cluster_id
            if len(neighbors) < minpts:
                continue
            seed_set.extend(neighbors)
    return labels


def cluster(db, labels):
    clusters = [[] for _ in range(max(labels)+1)]
    for p, l in tqdm(zip(db, labels), desc="clustering"):
        clusters[l].append(p)
    clusters.sort(key=lambda c: len(c), reverse=True)
    return clusters


def log(clusters, prefix, extension):
    for i, clstr in enumerate(clusters):
        file = f'{prefix}_cluster_{i}.{extension}'
        with open(file, mode='w') as f:
            for p in clstr:
                id = p[0]
                f.write(f'{id}\n')


if __name__ == '__main__':
    file, n, eps, minpts = parse()
    dataset = read_csv(file, '\t')
    labels = dbscan(dataset, eps, minpts, euclidean_distance)
    clusters = cluster(dataset, labels)[:n]
    prefix = os.path.basename(file).split('.')[0]
    log(clusters, prefix, 'txt')
