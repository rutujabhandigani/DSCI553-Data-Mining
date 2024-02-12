import os
import sys
import time
from itertools import combinations
from pyspark import SparkConf, SparkContext
from pyspark.sql import *
from graphframes import *

def read_data(sc, input_file_path):
  input_rdd = sc.textFile(input_file_path).filter(lambda x: not x.startswith("user_id")).map(lambda x: x.split(',')).cache()
  return input_rdd

def construct_graph(sqlContext, user_pair_list, user_bus_dict, filter_threshold):
  nodes, edges = set(), []
  filter_threshold = int(filter_threshold)
  for pair in user_pair_list:
      if len(user_bus_dict[pair[0]] & user_bus_dict[pair[1]])>= filter_threshold:
          nodes.add(pair[0])
          nodes.add(pair[1])
          edges.append((pair[0], pair[1]))
          edges.append((pair[1], pair[0]))
  nodes = list(nodes)

  #vertices = sqlContext.createDataFrame([(uid,) for uid in vertices_set], ["id"])
  #edges = sqlContext.createDataFrame(list(edges_set), ["src", "dst"])
  vertices_df = sqlContext.createDataFrame([tuple([vertices]) for vertices in nodes], ["id"])
  edges_df = sqlContext.createDataFrame(edges, ["src", "dst"])
  return GraphFrame(vertices_df, edges_df)

def write_output(community_groups, output_path):
  with open(output_path, 'w') as output_file:
      for i in community_groups.collect():
          output_file.write(str(i)[1:-1]+"\n")

def task1(filter_threshold, input_file_path, output_file_path):

  sc = SparkContext(appName= "task1")
  sc.setLogLevel('ERROR')
  sqlContext = SQLContext(sc)

  input_rdd = read_data(sc, input_file_path)

  user = input_rdd.map(lambda x:x[0]).distinct()
  user_total = len(user.collect())

  business = input_rdd.map(lambda x:x[1]).distinct()
  business_total = len(business.collect())
  user_bus_dict= input_rdd.map(lambda row : (row[0], row[1])).groupByKey().map(lambda x:(x[0],sorted(list(x[1])))).mapValues(set).collectAsMap()
  user_pair_list = list(combinations(user.collect(),2))

  graph = construct_graph(sqlContext, user_pair_list, user_bus_dict, filter_threshold)

  result = graph.labelPropagation(maxIter=5)
  community_groups = result.rdd.map(lambda x: (x[1],x[0])).groupByKey().map(lambda x: sorted(list(x[1]))).sortBy(lambda x:(x[0])).sortBy(lambda x :(len(x)))

  write_output(community_groups, output_file_path)

if __name__ == '__main__':
  filter_threshold = sys.argv[1]
  input_file_path = sys.argv[2]
  output_file_path = sys.argv[3]

  os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
  os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
  os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 pyspark-shell"

  start_time = time.time()
  task1(filter_threshold, input_file_path, output_file_path)

  print("Duration:", time.time() - start_time)