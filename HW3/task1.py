from pyspark import SparkContext
import random
import sys
import time
import os
import operator
from itertools import combinations
from collections import defaultdict


def load_data(input_path):
    rdd = sc.textFile(input_path)
    header = rdd.first()
    data = rdd.filter(lambda row: row != header).map(lambda row: row.split(","))
    return data

def get_business_user_dict(data):
    business_user = data.map(lambda row: (row[1], row[0])).groupByKey().mapValues(set)
    business_user_dict = {bus: users for bus, users in business_user.collect()}
    return business_user, business_user_dict

def get_users_dict(data):
    users = data.map(lambda row: row[0]).distinct().collect()
    return {user: i for i, user in enumerate(users)}

def get_minhash_signatures(data, users_dict, hash_size, m):
    sign_dict = {}
    a = random.sample(range(1, m), hash_size)
    b = random.sample(range(1, m), hash_size)
    hash_func = [a, b]

    for bus, user_list in data.collect():
        minhash_sign_list = []
        for i in range(hash_size):
            minhash = float("inf")
            for user in user_list:
                minhash = min(minhash, (((hash_func[0][i] * users_dict[user] + hash_func[1][i]) % 1e9 + 7) % m))
            minhash_sign_list.append(int(minhash))
        sign_dict[bus] = minhash_sign_list
    return sign_dict

def get_bands(sign_dict, bands, row):
    band_dict = defaultdict(list)
    for business, minhash_sign in sign_dict.items():
        for i in range(0, bands):
            idx = (i, tuple(minhash_sign[i * row: i * row + row]))
            band_dict[idx].append(business)
    bands_dict = {key: val for key, val in band_dict.items() if len(val) > 1}
    return bands_dict

def get_candidates(bands_dict):
    candidates = set()
    for values in bands_dict.values():
        combination_list = combinations(sorted(values), 2)
        for item in combination_list:
            candidates.add(item)
    return candidates

def get_jaccard_similarity(candidate, business_user_dict):
    business1, business2 = candidate
    user1 = business_user_dict[business1]
    user2 = business_user_dict[business2]
    jaccard_sim = len(user1 & user2) / len(user1 | user2)
    return [business1, business2, jaccard_sim]


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    #input_path = "yelp_train.csv"
    #output_path = "task1_op_rb.csv"

    os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel("ERROR")

    hash_size = 60
    bands = 30
    row = 2
    
    start_time = time.time()
    
    data = load_data(input_path)
    business_user, business_user_dict = get_business_user_dict(data)
    user_dict = get_users_dict(data)       
    m = len(user_dict)

    sign_dict = get_minhash_signatures(business_user, user_dict, hash_size, m)
    bands_dict = get_bands(sign_dict, bands, row)
    candidates = get_candidates(bands_dict)
 
    result = {}
    for candidate in candidates:
        similarity = get_jaccard_similarity(candidate, business_user_dict)
        if similarity[2] >= 0.5:
            result[str(similarity[0]) + "," + str(similarity[1])] = similarity[2]

    result = dict(sorted(result.items(), key=operator.itemgetter(0)))
    result_str = "business_id_1, business_id_2, similarity\n"
    for key, values in result.items():
        result_str += key + "," + str(values) + "\n"
    with open(output_path, "w") as f:
        f.writelines(result_str)

    duration = time.time() - start_time
    print('Duration: ', duration)