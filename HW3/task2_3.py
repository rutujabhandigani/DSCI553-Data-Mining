from pyspark import SparkContext
import json
import sys
import time
import numpy as np
from xgboost import XGBRegressor


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

def read_csv_data(path):
    lines = spark.textFile(path)
    header = lines.first()
    return lines.filter(lambda row: row != header).map(lambda row: row.split(","))

def parse_json(spark_context, file_path, map_func):
    return spark_context.textFile(file_path).map(lambda row: json.loads(row)).map(map_func)

def collect_to_dict(rdd, aggregation_func=None):
    if aggregation_func:
        rdd = rdd.groupByKey().mapValues(aggregation_func)
    return {key: value for key, value in rdd.collect()}

def compute_average(values):
    useful, funny, cool, num = 0, 0, 0, 0
    for u, f, c in values:
        useful += u
        funny += f
        cool += c
        num += 1
    return (useful / num, funny / num, cool / num) if num else (None, None, None)

def prepare_features(row, review_dict, user_dict, bus_dict):
    bus, user = row[1], row[0]
    review_features = review_dict.get(bus, (None, None, None))
    user_features = user_dict.get(user, (None, None, None))
    bus_features = bus_dict.get(bus, (None, None))
    return [*review_features, *user_features, *bus_features]

def model_based_main(spark, folder_path, val_path):

    review_dict = collect_to_dict(parse_json(spark, f'{folder_path}/review_train.json', lambda row: (row['business_id'], (float(row['useful']), float(row['funny']), float(row['cool'])))), compute_average)
    user_dict = collect_to_dict(parse_json(spark, f'{folder_path}/user.json', lambda row: (row['user_id'], (float(row['average_stars']), float(row['review_count']), float(row['fans'])))))
    bus_dict = collect_to_dict(parse_json(spark, f'{folder_path}/business.json', lambda row: (row['business_id'], (float(row['stars']), float(row['review_count'])))))

    lines_train = read_csv_data(f'{folder_path}/yelp_train.csv')
    lines_val = read_csv_data(val_path)

    X_train = np.array([prepare_features(row, review_dict, user_dict, bus_dict) for row in lines_train.collect()], dtype='float32')
    Y_train = np.array([float(row[2]) for row in lines_train.collect()], dtype='float32')

    user_bus_list = [(row[0], row[1]) for row in lines_val.collect()]
    X_val = np.array([prepare_features(row, review_dict, user_dict, bus_dict) for row in user_bus_list], dtype='float32')

    xgb_params = {
        'lambda': 9.92724463758443,
        'alpha': 0.2765119705933928,
        'colsample_bytree': 0.5,
        'subsample': 0.8,
        'learning_rate': 0.02,
        'max_depth': 17,
        'random_state': 2020,
        'min_child_weight': 101,
        'n_estimators': 300,
    }
    xgb = XGBRegressor(**xgb_params)
    xgb.fit(X_train, Y_train)
    Y_pred = xgb.predict(X_val)

    return user_bus_list, Y_pred


if __name__ == '__main__':
    folder_path = sys.argv[1]
    val_path = sys.argv[2]
    output_path = sys.argv[3]
    
    spark = SparkContext('local[*]', 'task2_3')
    spark.setLogLevel("ERROR")
    
    
    start_time = time.time()

    train_path = folder_path + '/yelp_train.csv'
    train_data = load_and_process_data(spark, train_path, is_train=True)
    validation_data = load_and_process_data(spark, val_path, is_train=False)

    bus_user_dict, user_bus_dict, bus_avg_dict, user_avg_dict, bus_user_r_dict = process_training_data(train_data)

    data_dicts = (bus_user_dict, user_bus_dict, bus_avg_dict, user_avg_dict, bus_user_r_dict)
    w_dict = {}

    item_based_result = []
    for row in validation_data.collect():
        prediction = item_based(row[0], row[1], data_dicts, w_dict)
        item_based_result.append(prediction)

    user_bus_list, model_based_result = model_based_main(spark, folder_path, val_path)

    factor = 0.1
    result_str = "user_id, business_id, prediction\n"
    for i in range(0, len(model_based_result)):
        result = float(factor) * float(item_based_result[i]) + (1 - float(factor)) * float(model_based_result[i])
        result_str += user_bus_list[i][0] + "," + user_bus_list[i][1] + "," + str(result) + "\n"
        
    with open(output_path, "w") as f:
        f.writelines(result_str)

    print('Duration: ', time.time() - start_time)