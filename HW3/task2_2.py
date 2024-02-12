from pyspark import SparkContext
import json
import sys
import time
import numpy as np
from xgboost import XGBRegressor

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

if __name__ == '__main__':
    folder_path = sys.argv[1]
    val_path = sys.argv[2]
    output_path = sys.argv[3]
    
    
    spark = SparkContext('local[*]', 'task2_2')
    spark.setLogLevel("ERROR")
    
    
    start_time = time.time()


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


    result_str = "user_id,business_id,prediction\n" + "\n".join([f"{user},{bus},{pred}" for (user, bus), pred in zip(user_bus_list, Y_pred)])
    with open(output_path, "w") as f:
        f.write(result_str)

    print('Duration:', time.time() - start_time)
