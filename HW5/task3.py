from blackbox import BlackBox
import time
import sys
import csv
import random


def write_output(result_str, output_path):
    with open(output_path, 'w') as f:
        f.writelines(result_str)
    f.close()

def main():
    #input_path = "users.txt"
    #stream_size = 100
    #num_of_asks = 30
    #output_path = "task3_output_rb.csv"

    input_path = sys.argv[1]
    stream_size = sys.argv[2]
    num_of_asks = sys.argv[3]
    output_path = sys.argv[4]
    
    start_time = time.time()
    bx = BlackBox()

    random.seed(553)
    users_list = []
    n = 0
    
    result_str = "seqnum,0_id,20_id,40_id,60_id,80_id\n"
    stream_size = int(stream_size)
    num_of_asks = int(num_of_asks)
    
    for i in range(num_of_asks):
        stream_users = bx.ask(input_path, stream_size)
        for user in stream_users:
            n = n + 1
            if len(users_list) < 100:
                users_list.append(user)

            elif random.random() < 100 / n:
                users_list[random.randint(0, 99)] = user

            if n % 100 == 0:
                result_str += str(n) + ',' + users_list[0] + ',' + users_list[20] + ',' + users_list[40] + ',' + users_list[60] + ',' + users_list[80] + '\n'
    
    write_output(result_str, output_path)

    print('Duration: ', time.time() - start_time)

if __name__ == '__main__':
    main()