from blackbox import BlackBox
import time
import sys
import random
import binascii
import csv


def fm_algo(input_path, stream_size, num_of_asks):
    bx = BlackBox()
    stream_size = int(stream_size)
    num_of_asks = int(num_of_asks)
    ground_truth_t = 0
    estimate_t = 0
    result_str = "Time,Ground Truth,Estimation\n"
    
    for i in range(num_of_asks):
        stream_users = bx.ask(input_path, stream_size)
        
        ground_truth, exist_hash = set(), []
        
        for user in stream_users:
            result = myhashs(user)
            ground_truth.add(user) if user not in ground_truth else None
            exist_hash.append(result)
        
        estimate_sum = 0
        for j in range(50):
            temp = []
            for value in exist_hash:
                temp.append(int(value[j]))
            max_t_zero = 0
            for value in temp:
                temp_str = bin(value)[2:]
                trailing_zeros = temp_str.rstrip('0')
                trailing_zeros_length = len(temp_str) - len(trailing_zeros)
                if trailing_zeros_length > max_t_zero:
                    max_t_zero = trailing_zeros_length
            estimate_sum += 2 ** max_t_zero
        estimate = estimate_sum // 50
        ground_truth_t += len(ground_truth)
        estimate_t += estimate
        result_str = result_str + str(i) + ',' + str(len(ground_truth)) + ',' + str(estimate) + '\n'
    
    result = estimate_t/ground_truth_t
    return result_str, result

def myhashs(user):
    pm = 10**9 + 7
    max_range = 997
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


def task2():
    #input_path = "users.txt"
    #stream_size = 100
    #num_of_asks = 30
    #output_path = "task2_output_rb.csv"

    input_path = sys.argv[1]
    stream_size = sys.argv[2]
    num_of_asks = sys.argv[3]
    output_path = sys.argv[4]
    
    start_time = time.time()
    
    result_str, result = fm_algo(input_path, stream_size, num_of_asks)
    write_output(result_str, output_path)

    print(result)
    print('Duration: ', time.time() - start_time)

if __name__ == '__main__':
    task2()