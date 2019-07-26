# 在提取进程event list

import json
import argparse
import os


# dictionary for running process
processTable = {
    'buffer': []
}

def get_session(jsonfile):
    with open(jsonfile) as infile:
        for line in infile:
            event = json.loads(line)
            event_id = event['Event']['System']['EventID']
            if(event_id == '4688'):
                add_process(event)
            elif(event_id == '4689'):
                end_process(event)
            elif(event_id == '4663'):
                access_file(event)
            

def add_process(event):
    global processTable

    processID = event['Event']['EventData']['NewProcessId']
    if(processID in processTable):
        print('Process exist')
        flush_process(processID)
    processTable[processID] = {
        'Name': event['Event']['EventData']['NewProcessName'],
        'Event': []
    }


def end_process(event):
    global processTable

    processID = event['Event']['EventData']['ProcessId']
    if(processID in processTable):
        flush_process(processID)


def flush_process(processID):
    global processTable

    p = processTable[processID]
    p["ProcessID"] = processID
    processTable['buffer'].append(p)
    del processTable[processID]


def access_file(event):
    global processTable

    processID = event['Event']['EventData']['ProcessId']
    if(processID not in processTable):
        return
    processName = event['Event']['EventData']['ProcessName']
    if(processName != processTable[processID]['Name']):
        print('Process Name Mismatch')
        return
    objectName = event['Event']['EventData']['ObjectName']
    processTable[processID]['Event'].append(objectName)


def aggregate_process():
    global processTable

    processTable['table'] = {}
    for p in processTable['buffer']:
        if(p['Name'] not in processTable['table']):
            processTable['table'][p['Name']] = []
        processTable['table'][p['Name']].append(p['Event'])


def dump_table(of):
    global processTable

    with open(of, 'w') as outfile:
        json.dump(processTable['table'], outfile, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dst_file', help='the event file which you want to analysis')
    args = parser.parse_args()

    get_session(args.dst_file)
    aggregate_process()
    dump_table(os.path.splitext(args.dst_file)[0] + '.dump.json')