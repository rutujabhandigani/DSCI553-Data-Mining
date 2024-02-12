from pyspark import SparkContext
from itertools import permutations, combinations
from copy import deepcopy
import sys
import time
from collections import defaultdict

def process_input_data(sc, input_path, threshold):
    lines = sc.textFile(input_path).filter(lambda x: not x.startswith("user_id")).map(lambda x: x.split(',')).cache()
    user_bus = lines.groupByKey().mapValues(set)
    user_bus_dict = {user: bus for user, bus in user_bus.collect()}
    users = lines.map(lambda row: row[0]).distinct()
    candidates = list(permutations(users.collect(),2))
    edges = [candidate for candidate in candidates if len(user_bus_dict[candidate[0]] & user_bus_dict[candidate[1]]) >= int(threshold)]
    v_list = list(set(candidate[0] for candidate in edges))

    graph = {}
    for user1, user2 in edges:
      graph.setdefault(user1, set()).add(user2)
      graph.setdefault(user2, set()).add(user1)

    return v_list, graph

def GN(graph, v_list):
    bet = defaultdict(float)
    for root in v_list:
        parent, level, n_shortest, path, queue, visitor = defaultdict(set), {}, defaultdict(float), [], [root], {root}
        level[root], n_shortest[root] = 0, 1

        while queue:
            current = queue.pop(0)
            path.append(current)
            for neighbor in graph[current]:
                if neighbor not in visitor:
                    queue.append(neighbor)
                    visitor.add(neighbor)
                    parent[neighbor].add(current)
                    n_shortest[neighbor] += n_shortest[current]
                    level[neighbor] = level[current] + 1
                elif level[neighbor] == level[current] + 1:
                    parent[neighbor].add(current)
                    n_shortest[neighbor] += n_shortest[current]

        v_w = {p: 1 for p in path}
        edge_w = defaultdict(float)

        for p in reversed(path):
            for q in parent[p]:
                temp_w = v_w[p] * (n_shortest[q] / n_shortest[p])
                v_w[q] += temp_w
                edge = tuple(sorted([p, q]))
                edge_w[edge] += temp_w

        for key, value in edge_w.items():
            bet[key] += value / 2

    return sorted(bet.items(), key=lambda x: (-x[1], x[0]))

def calculate_modularity(sub_graph, candidates, k, n_edge):
    modularity = 0.0
    for candidate in candidates:
        for p in candidate:
            for q in candidate:
                if q in sub_graph[p]:
                  a_p_q = 1.0
                else:
                  a_p_q = 0.0
                modularity += a_p_q - (k[p] * k[q]) / (2.0 * n_edge)
    return modularity / (2 * n_edge)

def update_graph_and_get_modularity(graph, betweenness_values, v_list):
    sub_graph = deepcopy(graph)
    n_edge = len(betweenness_values)
    k = {n: len(graph[n]) for n in graph}
    max_m = -float('inf')
    result = None

    while betweenness_values:
        candidates = []
        v_s = v_list.copy()
        for root in v_s:
            queue = [root]
            visitor = {root}
            while queue:
                current = queue.pop(0)
                for p in sub_graph[current]:
                    if p not in visitor:
                        v_s.remove(p)
                        queue.append(p)
                        visitor.add(p)
            candidates.append(sorted(list(visitor)))

        modularity = calculate_modularity(sub_graph, candidates, k, n_edge)
        if modularity > max_m:
            max_m = modularity
            result = deepcopy(candidates)

        highest_betweenness = betweenness_values[0][1]
        for v, value in betweenness_values:
            if value >= highest_betweenness:
                sub_graph[v[0]].remove(v[1])
                sub_graph[v[1]].remove(v[0])

        betweenness_values = GN(sub_graph, v_list)

    return sorted(result, key=lambda x: (len(x), x[0]))


def write_betweenness_to_file(betweenness_path, betweenness_values):
    with open(betweenness_path, "w") as file:
        for edge, value in betweenness_values:
            edge_str = str(edge)
            line = f"{edge_str},{round(value, 5)}\n"
            file.write(line)

def write_communities_to_file(output_path, communities):
    with open(output_path, "w") as file:
        for community in communities:
            community_str = str(community)
            line = community_str[1:-1] + "\n"
            file.write(line)

if __name__ == '__main__':
    threshold = sys.argv[1]
    input_path = sys.argv[2]
    between_path = sys.argv[3]
    output_path = sys.argv[4]

    #threshold = 5
    #input_path = "ub_sample_data.csv"
    #between_path = "between_rb4.txt"
    #output_path = "community_rb4.txt"

    sc = SparkContext('local[*]', 'task2')
    sc.setLogLevel('ERROR')

    start_time = time.time()
    
    v_list, graph = process_input_data(sc, input_path, threshold)
    betweenness_values = GN(graph, v_list)
    write_betweenness_to_file(between_path, betweenness_values)

    communities = update_graph_and_get_modularity(graph, betweenness_values, v_list)
    write_communities_to_file(output_path, communities)

    print('Duration: ', time.time() - start_time)