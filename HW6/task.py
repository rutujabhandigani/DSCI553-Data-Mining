import numpy as np
from sklearn.cluster import KMeans
import time
import sys


def read_data(input_path):
    with open(input_path, "r") as file:
        npdata = np.array([list(map(float, line.strip().split(','))) for line in file])

    return npdata

def split_and_cluster(npdata, n_cluster):
    np.random.shuffle(npdata)

    split_parts = 5
    npdata = np.array_split(npdata, split_parts)

    data1 = npdata[0]
    kmeans1 = KMeans(n_clusters=n_cluster).fit(data1[:, 2:])

    return npdata, data1, kmeans1

def create_clusters(kmeans1, n_cluster):
    clusters = {}
    for idx, cid in enumerate(kmeans1.labels_):
        clusters.setdefault(cid, []).append(idx)

    rs = {idx[0] for idx in clusters.values() if len(idx) == 1}

    return rs, clusters
    
def clusters_refit(data1, rs, n_cluster):    
    ds = np.delete(data1, list(rs), axis=0)

    clusters = {}
    kmeans2 = KMeans(n_clusters=n_cluster).fit(ds[:, 2:])
    clusters = {cid: [] for cid in range(n_cluster)}
    for idx, cid in enumerate(kmeans2.labels_):
        clusters[cid].append(idx)

    return ds, clusters


def calculate_cluster_statistics(data, indices):
    cluster_stats = {}
    cluster_centroid = {}
    cluster_deviation = {}
    cluster_points = {}

    for cid, idx in indices.items():
        cluster_data = data[idx, 2:]
        n = len(idx)
        SUM = np.sum(cluster_data, axis=0)
        SUMSQ = np.sum(np.square(cluster_data), axis=0)

        cluster_stats[cid] = [n, SUM, SUMSQ]
        cluster_points[cid] = data[idx, 0].astype(int).tolist()

        centroid = SUM / n
        cluster_centroid[cid] = centroid

        deviation = np.sqrt(np.subtract(SUMSQ / n, np.square(centroid)))
        cluster_deviation[cid] = deviation

    return cluster_stats, cluster_centroid, cluster_deviation, cluster_points

def mahalanobis_distance(point1, point2, deviation):
    return np.sqrt(np.sum(np.square(np.divide(np.subtract(point1, point2), deviation)), axis=0))

def calculate_mahalanobis_distance(point1, point2, deviation):
    deviation = np.where(deviation != 0, deviation, np.ones_like(deviation))
    distance = mahalanobis_distance(point1, point2, deviation)
    return distance

def merge_clusters(dict1, dict2, c_dict1, c_dict2, d_dict1, d_dict2, point_dict1, point_dict2, D):
    new_dict = {}
    for cid1 in dict1:
        closest_cluster, min_distance = -1, float('inf')
        for cid2 in dict2:
            if cid1 != cid2:
                distance1 = calculate_mahalanobis_distance(c_dict1[cid1], c_dict2[cid2], d_dict2[cid2])
                distance2 = calculate_mahalanobis_distance(c_dict2[cid2], c_dict1[cid1], d_dict1[cid1])
                distance = min(distance1, distance2)
                if distance < min_distance:
                    min_distance, closest_cluster = distance, cid2

        if min_distance < D:
            D = min_distance
            new_dict[cid1] = closest_cluster

    for cid1, cid2 in new_dict.items():
        if cid1 in dict1 and cid2 in dict2 and cid1 != cid2:
            n = dict1[cid1][0] + dict2[cid2][0]
            SUM = np.add(dict1[cid1][1], dict2[cid2][1])
            SUMSQ = np.add(dict1[cid1][2], dict2[cid2][2])

            centroid = SUM / n
            d = np.sqrt(np.subtract(SUMSQ / n, np.square(centroid)))

            dict2[cid2] = [n, SUM, SUMSQ]
            c_dict2[cid2] = centroid
            d_dict2[cid2] = d
            point_dict2[cid2].extend(point_dict1[cid1])

            del dict1[cid1]
            del c_dict1[cid1]
            del d_dict1[cid1]
            del point_dict1[cid1]

    return D

def find_closest_cluster(data_point, clusters_dict, centroids_dict, deviations_dict, threshold):
    min_distance = float('inf')
    closest_cluster = -1
    for cid, _ in clusters_dict.items():
        distance = mahalanobis_distance(data_point, centroids_dict[cid], deviations_dict[cid])
        if distance < min_distance:
            min_distance = distance
            closest_cluster = cid

    return closest_cluster if min_distance < threshold else -1

def update_cluster(cluster_id, data_point, clusters_dict, centroids_dict, deviations_dict):
    n, SUM, SUMSQ = clusters_dict[cluster_id]
    n += 1
    new_SUM = np.add(SUM, data_point)
    new_SUMSQ = np.add(SUMSQ, np.square(data_point))

    new_centroid = new_SUM / n
    new_deviation = np.sqrt(np.subtract(new_SUMSQ / n, np.square(new_centroid)))

    clusters_dict[cluster_id] = [n, new_SUM, new_SUMSQ]
    centroids_dict[cluster_id] = new_centroid
    deviations_dict[cluster_id] = new_deviation

def write_results(output_path, ds_dict, cs_dict, rs, round_num, append=False):
    mode = 'a' if append else 'w'
    with open(output_path, mode) as file:
        if not append:
            file.write('The intermediate results:\n')
        
        num_ds = sum(value[0] for value in ds_dict.values())
        num_cs = sum(value[0] for value in cs_dict.values())

        result_str = f'Round {round_num}: {num_ds},{len(cs_dict)},{num_cs},{len(rs)}\n'
        file.write(result_str)

def write_final_results(output_path, results):
    with open(output_path, "a") as file:
        file.write('\n')
        file.write('The clustering results:\n')
        for point in sorted(results.keys(), key=int):
            file.write(f'{point},{results[point]}\n')


def task(input_path, n_cluster, output_path):

    D = 0
    npdata = read_data(input_path)   
    npdata, data1, kmeans1 = split_and_cluster(npdata, n_cluster)
    rs, clusters = create_clusters(kmeans1, n_cluster)   
    ds, clusters = clusters_refit(data1, rs, n_cluster)
    
    ds_dict, ds_c_dict, ds_d_dict, ds_point_dict = calculate_cluster_statistics(ds, clusters)


    cs_dict = {}
    cs_c_dict = {}
    cs_d_dict = {}
    cs_point_dict = {}

    data_rs = data1[list(rs), :]
    if len(rs) >= 5 * n_cluster:
        kmeans_temp = KMeans(n_clusters=5 * n_cluster).fit(data_rs[:, 2:])
        temp_clusters = {cid: [] for cid in set(kmeans_temp.labels_)}
        for idx, cid in enumerate(kmeans_temp.labels_):
            temp_clusters[cid].append(idx)
        rs = {idx[0] for idx in temp_clusters.values() if len(idx) == 1}

        cs_dict, cs_c_dict, cs_d_dict, cs_point_dict = calculate_cluster_statistics(data_rs, temp_clusters)


    write_results(output_path, ds_dict, cs_dict, rs, round_num=1)
    
    D = 2 * np.sqrt(data1.shape[1] - 2) 

    for i in range(2, 6):
        for idx, value in enumerate(npdata[i - 1]):
          data_point = value[2:]

          closest_cluster = find_closest_cluster(data_point, ds_dict, ds_c_dict, ds_d_dict, D)
          if closest_cluster != -1:
              update_cluster(closest_cluster, data_point, ds_dict, ds_c_dict, ds_d_dict)
              ds_point_dict[closest_cluster].append(int(value[0]))
          else:
              closest_cluster = find_closest_cluster(data_point, cs_dict, cs_c_dict, cs_d_dict, D)
              if closest_cluster != -1:
                  update_cluster(closest_cluster, data_point, cs_dict, cs_c_dict, cs_d_dict)
                  cs_point_dict[closest_cluster].append(int(value[0]))
              else:
                  rs.add(idx)

        data_rs = npdata[i - 1][list(rs), :]
        if len(rs) >= 5 * n_cluster:
            kmeans_temp = KMeans(n_clusters=5 * n_cluster).fit(data_rs[:, 2:])
            rs, clusters = create_clusters(kmeans_temp, n_cluster)
            cs_dict, cs_c_dict, cs_d_dict, cs_point_dict = calculate_cluster_statistics(data_rs, clusters)

        if i == 5:
            D = merge_clusters(cs_dict, ds_dict, cs_c_dict, ds_c_dict, cs_d_dict, ds_d_dict, cs_point_dict, ds_point_dict, D)
        else:
            D = merge_clusters(cs_dict, cs_dict, cs_c_dict, cs_c_dict, cs_d_dict, cs_d_dict, cs_point_dict, cs_point_dict, D)
        write_results(output_path, ds_dict, cs_dict, rs, round_num=i, append=True)


    if len(rs) > 0:
        data_rs = npdata[4][list(rs), 0]
        rs = set([int(n) for n in data_rs])
    result = {}
    for cid in ds_dict.keys():
        for point in ds_point_dict[cid]:
            result[point] = cid
    for cid in cs_dict.keys():
        for point in cs_point_dict[cid]:
            result[point] = -1
    for point in rs:
        result[point] = -1


    write_final_results(output_path, result)          


#input_path = "hw6_clustering.txt"
#n_cluster = 10
#output_path = "hw6_output_rb.txt"
input_path = sys.argv[1]
n_cluster = sys.argv[2]
output_path = sys.argv[3]
    
n_cluster = int(n_cluster)
start_time = time.time()
task(input_path, n_cluster, output_path)
print('Duration: ', time.time() - start_time)