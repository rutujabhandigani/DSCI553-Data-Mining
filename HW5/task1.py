from blackbox import BlackBox
import time
import sys
import random
import binascii
import csv


def bloom_filter(input_path, stream_size, num_of_asks):
    bx = BlackBox()
    stream_size = int(stream_size)
    num_of_asks = int(num_of_asks)

    exist_user, exist_hash = set(), []
    result_str = "Time,FPR\n"
    
    for i in range(num_of_asks):
        stream_users = bx.ask(input_path, stream_size)
        fp = 0
        for user in stream_users:
            result = myhashs(user)
            if result in exist_hash:
                if user not in exist_user:
                    fp += 1
            exist_hash.append(result)
            exist_user.add(user)
        result_str = result_str + str(i) + ',' + str(fp / stream_size) + '\n'

    return result_str


def myhashs(user):
    pm = 10**9 + 7
    max_range = 69997
    num_hashes = 50

    a = random.sample(range(1, max_range), num_hashes)
    b = random.sample(range(1, max_range), num_hashes)

    user_int = int(binascii.hexlify(user.encode('utf8')), 16)

    hash_values = []
    for i in range(num_hashes):
        hash_val = ((a[i] * user_int + b[i]) % pm) % max_range
        hash_values.append(hash_val)

    return hash_values

def write_output(result_str, output_path):
    with open(output_path, 'w') as f:
        f.writelines(result_str)
    f.close()

def task1():
    #input_path = "users.txt"
    #stream_size = 100
    #num_of_asks = 30
    #output_path = "task1_output_rb.csv"

    input_path = sys.argv[1]
    stream_size = sys.argv[2]
    num_of_asks = sys.argv[3]
    output_path = sys.argv[4]
    
    start_time = time.time()
    
    result_str = bloom_filter(input_path, stream_size, num_of_asks)
    write_output(result_str, output_path)
    
    print('Duration: ', time.time() - start_time)

if __name__ == '__main__':
    task1()