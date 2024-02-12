# Method Description:
#To enhance the recommendation system's effectiveness, I opted for a weighted hybrid recommendation system using model based collaborative filtering and #item based collaborative filtering. In the previous iteration of the recommendation system, I had used XGBoost Regressor with a set of parameters that #worked best based on trial and error. This time, I tried to further tune the hyperparameters using GridsearchCV, but the memory was running out on #vocareum.
#The next approach that I took was to integrate an additional attribute ‘wifi’ from business.json to map different 'WiFi' availability statuses ('free', #'no') to numerical values. While this improved the performance, but I still couldn’t beat the RMSE of .9800 (my RMSE value was 0.9803953370583288). 
#But I realized that enriching the recommendation process by using additional features was the way to go. The "attributes" dictionary in business.json had #various subcategories like restaurant's price range, credit card acceptance, takeout availability, delivery services, table reservations, and suitability #for breakfast, lunch, and dinner. Incorporating these features into the dataset intuitively enhanced the recommendation process by considering these #additional features, potentially leading to more refined and personalized recommendations based on users' preferences, and resulted in a notable reduction #in RMSE.
#==================================================
#Error Distribution:
#>=0 and <1: 102179
#>=1 and <2: 32919
#>=2 and <3: 6107
#>=3 and <4: 838
#>=4: 1
#==================================================
#Duration:  520.0189783573151
#==================================================
#RMSE (val): 0.9792808581182428
#==================================================
#RMSE (test): 0.9777198976510273
#==================================================

from pyspark import SparkContext
import xgboost as xgb
import pandas as pd
import math
import json
import random
import sys
import time
from collections import defaultdict
from itertools import combinations

weights_dict = {}

def pearsonSimilarity(pair,ratings):
    try:
        item1_users = set(ratings[pair[0]].keys())
        items2_users = set(ratings[pair[1]].keys())

        co_rated_users = set(item1_users) & set(items2_users)
        if(len(co_rated_users)>9):
            item1_ratings = [float(ratings[pair[0]][user]) for user in co_rated_users]
            item2_ratings = [float(ratings[pair[1]][user]) for user in co_rated_users]

            item1_average = sum(item1_ratings)/len(item1_ratings)
            item2_average = sum(item2_ratings)/len(item2_ratings)

            item1_rating_average = list(map(lambda rating: rating - item1_average, item1_ratings))
            item2_rating_average = list(map(lambda rating: rating - item2_average, item2_ratings))

            numerator = sum(item1_rating_average * item2_rating_average)
            denominator = math.sqrt(sum(item1_rating_average * item1_rating_average)) * math.sqrt(sum(item2_rating_average * item2_rating_average))

            if numerator==0 or denominator==0:
                return 0.5
            else:
                return numerator/denominator

        else:
            return 0.5
    except:
        return 0.5

def getWeight(test_business_id, train_business_id, ratings):
    key1 = str(test_business_id) + "_" + str(train_business_id)
    key2 = str(train_business_id) + "_" + str(test_business_id)

    weight = weights_dict.get(key1)
    if weight is None:
        weight = weights_dict.get(key2)
        if weight is None:
            weight = pearsonSimilarity((test_business_id, train_business_id), ratings)
            weight = weight * pow(abs(weight), 2)
            weights_dict[key1] = weight

    return weight

def process_train_file(file_path):
    file = sc.textFile(file_path)
    header = file.first()
    train_reviews = file.filter(lambda x: x != header).map(lambda x: x.split(",")).map(lambda x: (x[1], x[0], x[2])).distinct().persist()
    return train_reviews

def create_businessID_userID_list(train_reviews):

    to_list = lambda a: [a]
    append = lambda a, b: a + [b]
    extend = lambda a, b: a + b

    businessID_userID_list = train_reviews.map(lambda x: (x[0], x[1])).combineByKey(to_list, append, extend).persist()
    filteredBusinesses = businessID_userID_list.map(lambda x: x[0]).distinct().collect()
    return businessID_userID_list, filteredBusinesses

def create_ratings(train_reviews):

    to_list = lambda a: [a]
    append = lambda a, b: a + [b]
    extend = lambda a, b: a + b

    ratings = train_reviews.map(lambda x: (x[0], (x[1], x[2]))).combineByKey(to_list, append, extend).map(lambda x: (x[0], dict(list(set(x[1]))))).collectAsMap()
    return ratings

def process_test_file(test_file, train_reviews):
    file = sc.textFile(test_file)
    header_test = file.first()

    to_list = lambda a: [a]
    append = lambda a, b: a + [b]
    extend = lambda a, b: a + b

    test_businessID_userID_list = file.filter(lambda x: x != header_test).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1])).zipWithIndex().persist()
    userID_businessID_dict = train_reviews.map(lambda x: (x[1], x[0])).combineByKey(to_list, append, extend).map(lambda x: (x[0], set(x[1]))).persist().collectAsMap()
    return test_businessID_userID_list, userID_businessID_dict

def calculate_predicted_ratings(test_businessID_userID_list, userID_businessID_dict, ratings, neighborhood):
    def sort_function(x):
        sorted_x = sorted(x, key=lambda y: y[1], reverse=True)
        return sorted_x[:min(len(sorted_x), neighborhood)]


    predicted_ratings = test_businessID_userID_list.map(lambda x: ((x[0][0], x[0][1], x[1]), userID_businessID_dict[x[0][0]])).map(lambda x: [(x[0], y) for y in x[1]]).flatMap(lambda x: x).map(lambda x: (x[0], (ratings[x[1]][x[0][0]], getWeight(x[0][1], x[1], ratings)))).groupByKey().mapValues(sort_function).map(lambda x: (x[0], sum([w * float(r) for r, w in x[1]]) / sum([abs(w) for _, w in x[1]]))).collect()
    predicted_ratings = sorted(predicted_ratings, key=lambda x: (x[0][2]))
    return predicted_ratings

def itemBased():
    #folder_path = "/content/gdrive/MyDrive/CompetitionStudentData/"
    folder_path = sys.argv[1]
    train_file_path = folder_path + '/yelp_train.csv'
    #test_file_path = "yelp_val.csv"
    test_file = sys.argv[2]
    neighborhood = 100

    train_reviews = process_train_file(train_file_path)
    businessID_userID_list, filteredBusinesses = create_businessID_userID_list(train_reviews)
    ratings = create_ratings(train_reviews)
    test_businessID_userID_list, userID_businessID_dict = process_test_file(test_file, train_reviews)
    predicted_ratings = calculate_predicted_ratings(test_businessID_userID_list, userID_businessID_dict, ratings, neighborhood)

    return predicted_ratings


def write_item_based_op(predicted_ratings, item_based_ratings_file):
    with open(item_based_ratings_file, "w") as f:
        f.write("user_id,business_id,prediction\n")
        for entry in predicted_ratings:
            f.write(f"{entry[0][0]},{entry[0][1]},{entry[1]}\n")


def returnBool(value):
    return 1.0 if value == 'True' else 0.0

def getAttribute(x, attribute):
    try:
        if attribute in x.keys():
            return returnBool(x[attribute])
        else:
            return 0.0
    except:
        return 0.0

def getPriceRange(x):
    if isinstance(x, dict):
        return float(x.get('RestaurantsPriceRange2', 2.0))
    return 2.0

def getAccpetedCards(x):
    return getAttribute(x, 'BusinessAcceptsCreditCards')

def getTakeout(x):
    return getAttribute(x, 'RestaurantsTakeOut')

def getReservations(x):
    return getAttribute(x, 'RestaurantsReservations')

def getDelivery(x):
    return getAttribute(x, 'RestaurantsDelivery')

def getBreakFast(x):
    meal_attr = getAttribute(x, 'GoodForMeal')
    return 1.0 if meal_attr and "'breakfast': True" in meal_attr else 0.0

def getLunch(x):
    meal_attr = getAttribute(x, 'GoodForMeal')
    return 1.0 if meal_attr and "'lunch': True" in meal_attr else 0.0

def getDinner(x):
    meal_attr = getAttribute(x, 'GoodForMeal')
    return 1.0 if meal_attr and "'dinner': True" in meal_attr else 0.0

def getBrunch(x):
    meal_attr = getAttribute(x, 'GoodForMeal')
    return 1.0 if meal_attr and "'brunch': True" in meal_attr else 0.0

def getWheelchairAccessible(x):
    return getAttribute(x, 'WheelchairAccessible')

def getOutdoorSeating(x):
    return getAttribute(x, 'OutdoorSeating')

def getHasTV(x):
    return getAttribute(x, 'HasTV')

def load_data(folder_path):

    user_sc = sc.textFile(folder_path+"/user.json").map(json.loads).map(lambda x: (x['user_id'],x['average_stars'],x['review_count'],x['useful'],x['fans'], x['compliment_note'], x['compliment_hot'])).collect()

    busniness_sc = sc.textFile(folder_path+"/business.json").map(json.loads).map(lambda x:(x['business_id'],x['stars'],x['review_count'],x['is_open'],x['attributes'])).collect()

    user_df = pd.DataFrame(user_sc, columns = ['user_id','average_stars','review_count','useful','fans', 'compliment_note', 'compliment_hot'])
    busniness_df = pd.DataFrame(busniness_sc, columns = ['business_id','stars','review_count','is_open','attributes'])

    busniness_df['PriceRange'] = busniness_df['attributes'].apply(lambda x: getPriceRange(x))
    busniness_df['CardAccepted'] = busniness_df['attributes'].apply(lambda x: getAccpetedCards(x))
    busniness_df['Takeout'] = busniness_df['attributes'].apply(lambda x: getTakeout(x))
    busniness_df['Reservations'] = busniness_df['attributes'].apply(lambda x: getReservations(x))
    busniness_df['Delivery'] = busniness_df['attributes'].apply(lambda x: getDelivery(x))
    busniness_df['Breakfast'] = busniness_df['attributes'].apply(lambda x: getBreakFast(x))
    busniness_df['Lunch'] = busniness_df['attributes'].apply(lambda x: getLunch(x))
    busniness_df['Dinner'] = busniness_df['attributes'].apply(lambda x: getDinner(x))
    busniness_df['Brunch'] = busniness_df['attributes'].apply(lambda x: getBrunch(x))
    busniness_df['WheelchairAccessible'] = busniness_df['attributes'].apply(lambda x: getWheelchairAccessible(x))
    busniness_df['OutdoorSeating'] = busniness_df['attributes'].apply(lambda x: getOutdoorSeating(x))
    busniness_df['HasTV'] = busniness_df['attributes'].apply(lambda x: getHasTV(x))

    busniness_df = busniness_df[['business_id', 'stars', 'review_count', 'is_open', 'PriceRange','CardAccepted', 'Takeout', 'Reservations', 'Delivery', 'Breakfast', 'Lunch', 'Dinner', 'Brunch', 'WheelchairAccessible','OutdoorSeating','HasTV']]

    user_df = user_df.rename({'review_count':'user_review_count'}, axis=1)
    busniness_df = busniness_df.rename({'stars':'business_stars','review_count':'business_review_count'}, axis=1)

    return user_df, busniness_df


def preprocess_data(train_df, user_df, busniness_df):
    train_df = pd.merge(train_df, user_df, on='user_id', how='inner')
    train_df = pd.merge(train_df, busniness_df, on='business_id', how='inner')

    user_rating_info = defaultdict(dict)
    for index, row in train_df.iterrows():
        if row['user_id'] not in user_rating_info.keys():
            user_rating_info[row['user_id']] = {}
            user_rating_info[row['user_id']]['std_sum'] = pow(row['stars'] - row['average_stars'],2)
            user_rating_info[row['user_id']]['min'] = row['stars']
            user_rating_info[row['user_id']]['max'] = row['stars']
            user_rating_info[row['user_id']]['no_of_reviews'] = 1
        else:
            user_rating_info[row['user_id']]['std_sum'] = user_rating_info[row['user_id']]['std_sum']+pow(row['stars'] - row['average_stars'],2)
            user_rating_info[row['user_id']]['min'] = row['stars'] if row['stars']<user_rating_info[row['user_id']]['min'] else user_rating_info[row['user_id']]['min']
            user_rating_info[row['user_id']]['max']= row['stars'] if row['stars']>user_rating_info[row['user_id']]['max'] else user_rating_info[row['user_id']]['max']
            user_rating_info[row['user_id']]['no_of_reviews'] += 1

    for key,val in user_rating_info.items():
        user_rating_info[key]['std'] = user_rating_info[key]['std_sum'] / user_rating_info[key]['no_of_reviews']

    user_rating_df = pd.DataFrame.from_dict(user_rating_info, orient='index')
    user_rating_df.index.name = 'user_id'
    train_df = pd.merge(train_df, user_rating_df, on='user_id', how='inner')
    train_df = train_df.drop(['std_sum', 'no_of_reviews'], axis=1)

    return train_df, user_rating_df


def train_model(train_df):
    train_df_features = train_df[['average_stars', 'user_review_count', 'useful', 'fans', 'business_stars', 'business_review_count', 'compliment_note', 'compliment_hot', 'PriceRange', 'CardAccepted', 'Takeout', 'Reservations', 'Delivery', 'Breakfast', 'Lunch', 'Dinner', 'OutdoorSeating', 'HasTV']]
    train_df_target = train_df[['stars']]
    
    param = {
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
    model = xgb.XGBRegressor(**param)
    model.fit(train_df_features, train_df_target)

    return model

def predict_ratings(test_df, user_df, busniness_df, user_rating_df, trained_model):
    test_df = pd.merge(test_df,user_df,on='user_id',how='left')
    test_df = pd.merge(test_df,busniness_df,on='business_id',how='left')
    test_df = pd.merge(test_df,user_rating_df,on='user_id', how='left')
    test_df = test_df.drop(['std_sum', 'no_of_reviews'], axis=1)

    test_users_id = test_df['user_id']
    test_business_id = test_df['business_id']

    test_df = test_df.drop(['user_id', 'business_id'], axis=1)
    test_df = test_df[['average_stars', 'user_review_count', 'useful', 'fans', 'business_stars', 'business_review_count', 'compliment_note', 'compliment_hot', 'PriceRange', 'CardAccepted', 'Takeout', 'Reservations', 'Delivery', 'Breakfast', 'Lunch', 'Dinner', 'OutdoorSeating', 'HasTV']]

    rating_prediction = trained_model.predict(test_df)

    data = {
    'user_id': test_users_id,
    'business_id': test_business_id,
    'prediction': rating_prediction
    }
    predicted_df = pd.DataFrame(data)

    return predicted_df


def modelBased():
    folder_path = sys.argv[1]
    test_file = sys.argv[2]

    #folder_path = "/content/gdrive/MyDrive/CompetitionStudentData/"
    #test_file = "yelp_val.csv"
    
    user_df, busniness_df = load_data(folder_path)
    
    train_df = pd.read_csv(folder_path+'/yelp_train.csv')
    train_df, user_rating_df = preprocess_data(train_df, user_df, busniness_df)
    
    model = train_model(train_df)
    
    test_df = pd.read_csv(test_file)
    
    predicted_df = predict_ratings(test_df, user_df, busniness_df, user_rating_df, model)
    
    return predicted_df


def write_model_based_op(predicted_df, model_based_ratings_file):
    with open(model_based_ratings_file, "w") as f:
        f.write("user_id,business_id,prediction\n")
        for _, row in predicted_df.iterrows():
            f.write(f"{row['user_id']},{row['business_id']},{str(row['prediction'])}\n")


def combine_ratings(item_based_ratings_file, model_based_ratings_file, output_file, alpha):
    alpha = float(alpha)
    with open(item_based_ratings_file, 'r') as item_based_ratings, open(model_based_ratings_file, 'r') as model_based_ratings, open(output_file, 'w') as output_file:
        output_file.write("user_id,business_id,prediction\n")

        item_based_ratings.readline()  
        model_based_ratings.readline()  

        item_based_rating = item_based_ratings.readline()
        model_based_rating = model_based_ratings.readline()

        while item_based_rating and model_based_rating:
            user_id, business_id, item_rating = item_based_rating.split(',')
            _, _, model_rating = model_based_rating.split(',')

            final_rating = (alpha * float(item_rating)) + ((1 - alpha) * float(model_rating))
            output_file.write(f"{user_id},{business_id},{str(final_rating)}\n")

            item_based_rating = item_based_ratings.readline()
            model_based_rating = model_based_ratings.readline()


        item_based_ratings.close()
        model_based_ratings.close()


if __name__=='__main__':
    folder_path = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]
    #folder_path = "/content/gdrive/MyDrive/CompetitionStudentData/"
    #test_file = "yelp_val.csv"
    #output_file = "competition_op.csv"

    sc= SparkContext('local[*]','competition')
    sc.setLogLevel("ERROR")

    start_time = time.time()

    predicted_df = modelBased()
    model_based_ratings_file = "model_based.csv"
    write_model_based_op(predicted_df, model_based_ratings_file)

    predicted_ratings = itemBased()
    item_based_ratings_file = "item_based.csv"
    write_item_based_op(predicted_ratings, item_based_ratings_file)

    alpha = 0.001
    combine_ratings(item_based_ratings_file, model_based_ratings_file, output_file, alpha)

    print(time.time()-start_time)