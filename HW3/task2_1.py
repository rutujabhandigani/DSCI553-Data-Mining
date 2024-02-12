from pyspark import SparkContext
import time
import os
import sys


def load_and_process_data(spark_context, file_path, is_train=True):

    data = spark_context.textFile(file_path)
    header = data.first() 
    data = data.filter(lambda row: row != header).map(lambda row: row.split(","))

    if is_train:
        data = data.map(lambda row: (row[1], row[0], row[2]))
    else:
        data = data.map(lambda row: (row[1], row[0]))
    return data

def process_training_data(train_data):
    bus_user_train = train_data.map(lambda row: (row[0], row[1])).groupByKey().mapValues(set)
    bus_user_dict = {bus: users for bus, users in bus_user_train.collect()}

    user_bus_train = train_data.map(lambda row: (row[1], row[0])).groupByKey().mapValues(set)
    user_bus_dict = {user: buses for user, buses in user_bus_train.collect()}

    bus_avg = train_data.map(lambda row: (row[0], float(row[2]))).groupByKey().mapValues(list).map(lambda x: (x[0], sum(x[1]) / len(x[1])))
    bus_avg_dict = {bus: rating for bus, rating in bus_avg.collect()}

    user_avg = train_data.map(lambda row: (row[1], float(row[2]))).groupByKey().mapValues(list).map(lambda x: (x[0], sum(x[1]) / len(x[1])))
    user_avg_dict = {user: rating for user, rating in user_avg.collect()}

    bus_user_r = train_data.map(lambda row: (row[0], (row[1], row[2]))).groupByKey().mapValues(set)
    bus_user_r_dict = {bus: {user_r[0]: user_r[1] for user_r in user_r_set} for bus, user_r_set in bus_user_r.collect()}

    return bus_user_dict, user_bus_dict, bus_avg_dict, user_avg_dict, bus_user_r_dict

def calculate_weight(bus, bus1, user_inter, bus_user_r_dict, bus_avg_dict):
    if len(user_inter) <= 1:
        return simple_weight(bus, bus1, bus_avg_dict)
    elif len(user_inter) == 2:
        return pairwise_weight(bus, bus1, user_inter, bus_user_r_dict)
    else:
        return complex_weight(bus, bus1, user_inter, bus_user_r_dict)

def simple_weight(bus, bus1, bus_avg_dict):
    return (5.0 - abs(bus_avg_dict[bus] - bus_avg_dict[bus1])) / 5

def pairwise_weight(bus, bus1, user_inter, bus_user_r_dict):
    user_inter = list(user_inter)
    weights = [
        (5.0 - abs(float(bus_user_r_dict[bus][user]) - float(bus_user_r_dict[bus1][user]))) / 5
        for user in user_inter
    ]
    return sum(weights) / len(weights)

def complex_weight(bus, bus1, user_inter, bus_user_r_dict):
    r1 = [float(bus_user_r_dict[bus][user]) for user in user_inter]
    r2 = [float(bus_user_r_dict[bus1][user]) for user in user_inter]
    avg1, avg2 = sum(r1) / len(r1), sum(r2) / len(r2)
    temp1, temp2 = [x - avg1 for x in r1], [x - avg2 for x in r2]
    X = sum(x * y for x, y in zip(temp1, temp2))
    Y = (sum(x ** 2 for x in temp1) ** 0.5) * (sum(x ** 2 for x in temp2) ** 0.5)
    return 0 if Y == 0 else X / Y

def build_weight_list(user, bus, user_bus_dict, bus_user_r_dict, w_dict, bus_avg_dict):
    w_list = []
    for bus1 in user_bus_dict[user]:
        temp = tuple(sorted((bus1, bus)))
        w = w_dict.get(temp)
        if w is None:
            user_inter = bus_user_dict[bus] & bus_user_dict[bus1]
            w = calculate_weight(bus, bus1, user_inter, bus_user_r_dict, bus_avg_dict)
            w_dict[temp] = w
        w_list.append((w, float(bus_user_r_dict[bus1][user])))
    return w_list


def item_based(bus, user, data_dicts, w_dict):
    bus_user_dict, user_bus_dict, bus_avg_dict, user_avg_dict, bus_user_r_dict = data_dicts

    default_rating = 3.5
    if user not in user_bus_dict.keys():
        return default_rating
    if bus not in bus_user_dict.keys():
        return user_avg_dict[user]

    w_list = build_weight_list(user, bus, user_bus_dict, bus_user_r_dict, w_dict, bus_avg_dict)   

    w_list_can = sorted(w_list, key=lambda x: -x[0])[:15]
    X = sum(w * r for w, r in w_list_can)
    Y = sum(abs(w) for w, r in w_list_can)
    return X / Y if Y != 0 else 3.5


def predict_ratings(validation_data, data_dicts, w_dict):
    result_str = "user_id, business_id, prediction\n"
    for row in validation_data.collect():
        prediction = item_based(row[0], row[1], data_dicts, w_dict)
        result_str += row[1] + "," + row[0] + "," + str(prediction) + "\n"
    return result_str

def write_output(output, file_path):
    with open(file_path, "w") as f:
        f.writelines(output)

if __name__ == '__main__':
    #train_path = "yelp_train.csv"
    #val_path = "yelp_val.csv"
    #output_path = "task2_op_rb.csv"

    train_path = sys.argv[1]
    val_path = sys.argv[2]
    output_path = sys.argv[3]

    os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

    start_time = time.time()

    sc = SparkContext('local[*]', 'task2_1')
    sc.setLogLevel("ERROR")

    train_data = load_and_process_data(sc, train_path, is_train=True)
    validation_data = load_and_process_data(sc, val_path, is_train=False)

    bus_user_dict, user_bus_dict, bus_avg_dict, user_avg_dict, bus_user_r_dict = process_training_data(train_data)

    data_dicts = (bus_user_dict, user_bus_dict, bus_avg_dict, user_avg_dict, bus_user_r_dict)
    w_dict = {}

    result_str = predict_ratings(validation_data, data_dicts, w_dict)
    write_output(result_str, output_path)

    end_time = time.time()
    print('Duration: ', end_time - start_time)
