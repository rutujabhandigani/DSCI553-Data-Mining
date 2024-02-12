# -*- coding: utf-8 -*-
"""task3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18_I3iYXPb0MDSEy-R_qprzXwC0vUJwHb
"""

import json
from pyspark import SparkContext
import os
import sys
import time

#review_path = sys.argv[1]
#business_path = sys.argv[2]
#output_path_a = sys.argv[3]
#output_path_b = sys.argv[4]
review_path = "test_review.json"
business_path = "business.json"
output_path_a = "output3a.txt"
output_path_b = "output3b.json"

# configuration on local machine
#os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
#os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'


sc = SparkContext('local[*]', 'task3')


review_rdd = sc.textFile(review_path).map(lambda line: json.loads(line))
business_rdd = sc.textFile(business_path).map(lambda line: json.loads(line))

review = sc.textFile(review_path).map(json.loads).map(lambda x: (x['business_id'], x['stars']))
business = sc.textFile(business_path).map(json.loads).map(lambda x: (x['business_id'], x['city']))

data = review.join(business).map(lambda x: (x[1][1], x[1][0])).groupByKey().map(lambda x: (x[0], sum(x[1]) / len(x[1])))


data_temp = (
    review_rdd
    .map(lambda review: (review["business_id"], review["stars"]))
    .join(business_rdd.map(lambda business: (business["business_id"], business["city"])))
    .values()  # Get the (stars, city) pairs
    .groupByKey()
    .mapValues(lambda stars: sum(stars) / len(stars))  # Calculate average stars for each city
)

#data_a = sorted(data.collect(), key=lambda x: (-x[1], x[0]))


m1_start = time.time()
data_2 = data.collect()
data_2.sort(key=lambda x: (-x[1], x[0]))

print([s for s in data_2[:10]])
m1_time = time.time() - m1_start

with open(output_path_a, "w") as output_file_a:
  output_file_a.write("city,stars\n")
  #for city, stars in data.sortBy(lambda x: (-x[1], x[0])).collect():
        #output_file_a.write(f"{city},{stars}\n")
  for line in data_2:
    output_file_a.write(line[0]+","+str(line[1])+"\n")

m2_start = time.time()
print(data.takeOrdered(10, key=lambda x: (-x[1], x[0])))
m2_time = time.time() - m2_start

output = {'m1': m1_time,
          'm2': m2_time,
          'reason': "Sorting data in python returns a part of the RDD, whereas using spark the entire RDD is returened which is a more expensive operation"
}

with open(output_path_b, 'w') as output_file_b:
  json.dump(output, output_file_b)

sc.stop()