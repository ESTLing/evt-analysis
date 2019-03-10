# This script will be used to statistic different of aspect of host process action

import json


def statis_non_empty_process():
    """ rst: 
    Recorded process number:  4105
    Non-empty process number:  329

    so lager part of processes was only leave their create log, 
    no more meaningful record for them
    """
    processes = json.load(open('intermediate/actions.json'))
    print('Recorded process number: ', len(processes))
    non_empty = 0
    for process in processes:
        if len(process['actions']) > 0:
            non_empty += 1
    print('Non-empty process number: ', non_empty)


def statis_same_name_process():
    """
    """
    process_dict = {}
    processes = json.load(open('intermediate/actions.json'))
    for process in processes:
        if len(process['actions']) == 0: continue
        name = process['image']
        if name in process_dict:
            process_dict[name] += 1
        else:
            process_dict[name] = 1
    for process in process_dict:
        print(process.ljust(100), process_dict[process])


if __name__ == "__main__":
    # statis_non_empty_process()
    statis_same_name_process()