# -*- coding: utf-8 -*-
'''Read evtx file and save as json file.

This script is used for process raw evtx files, it take directory as input,
this input directory should struct like this:

- evtx-data
    - 20190601
        - Bing
        - Gellar
    - ...

evtxdump cmd tool is required to do the real job, more infomation about this
tool can be found here: https://github.com/0xrawsec/golang-evtx
'''

import argparse
import json
from collections import OrderedDict

import Evtx.Evtx as evtx
import xmltodict
import os
import subprocess


def parseRecord(record):
    ''' take raw record as input, and return a python dict
    '''
    record = xmltodict.parse(record)

    if record.get('Event') is None or \
        record.get('Event').get('System') is None:
        print('Event error')
        return None
    if record.get('Event').get('EventData') is None or \
        record.get('Event').get('EventData').get('Data') is None:
        return record

    data_vals = {}
    data = record.get('Event').get('EventData').get('Data')
    if isinstance(data, list):
        for dataitem in data:
            try:
                data_vals[str(dataitem.get('@Name'))] = str(dataitem.get('#text'))
            except:
                pass
    elif isinstance(data, OrderedDict):
        try:
            data_vals[str(data.get('@Name'))] = str(data.get('#text'))
        except:
            pass
    else:
        print('Unexpect event type: ' + str(data))
    record['Event']['EventData'] = data_vals
    return record


def parseEvtx(file):
    ''' take file name as input, yield every record as dict
    '''
    with evtx.Evtx(file) as f:
        for record in f.records():
            yield parseRecord(record.xml())


def evtx2json(evtxfile, jsonfile):
    ''' Convert evtx format file into json format
    '''
    with open(jsonfile, 'w') as outfile:
        for event in parseEvtx(evtxfile):
            if event is None: continue
            json.dump(event, outfile, ensure_ascii=False)
            outfile.write('\n')


def evtx2json_cmd(evtxfile, jsonfile):
    ''' Convert evtx format file into json format using evtx cmd tool
    '''
    logstart, logend = logRanger(jsonfile)
    if(logstart is None):
        print('%s: Create new file' % jsonfile)
    else:
        print('%s:[ %s , %s ]' % (jsonfile, logstart, logend))
    evtxdump = '~/go/bin/evtxdump'
    cmd = evtxdump + ' -c -t ' + evtxfile
    cmd += '| sort -t: -k1 '
    cmd += '| awk \'{ print substr($0,index($0,":")+2) }\' '
    _logend = None
    with open(jsonfile, 'a') as outfile:
        try:
            proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, encoding='utf-8')
            for line in proc.stdout.split('\n'):
                # empty line reached, means all output are parsed
                if(not line): continue
                event = json.loads(line)
                _logend = event['Event']['System']['TimeCreated']['SystemTime']
                if(logend is None or _logend > logend):
                    outfile.write(line)
                    outfile.write('\n')
        except subprocess.SubprocessError:
            print(cmd)
    print('New End Time: %s\n' % _logend)


def evtxDir(folder, result):
    ''' Iterate directory, process every evtx file and store result
    '''
    if(not os.path.exists(result)):
        os.mkdir(result)
    if(not os.path.isdir(result)):
        print('%s is not a valid directory.' % result)
        exit(1)
    userlist = os.listdir(folder)
    for user in userlist:
        target_path = os.path.join(result, user)
        if(not os.path.exists(target_path)):
            os.mkdir(target_path)
        user = os.path.join(folder, user)
        loglist = os.listdir(user)
        for log in loglist:
            src_file = os.path.join(user, log)
            if(not log.endswith('.evtx') or not os.path.isfile(src_file)):
                continue
            log = log.split('.')[0] + '.json'
            dst_file = os.path.join(target_path, log)
            evtx2json_cmd(src_file, dst_file)


def logRanger(jsonfile):
    ''' return start time and end time of log file
    '''
    startTime = None
    endTime = None
    if(os.path.isfile(jsonfile)):
        with open(jsonfile) as infile:
            for line in infile:
                try:
                    event = json.loads(line)
                    if startTime is None:
                        startTime = event['Event']['System']['TimeCreated']['SystemTime']
                    endTime = event['Event']['System']['TimeCreated']['SystemTime']
                except:
                    pass
    return startTime, endTime


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='Result will store in this dirctory')
    parser.add_argument('-f', help='Target evtx file')
    parser.add_argument('-d', help='Iterate through the directory')
    args = parser.parse_args()
    if(args.d is not None):
        evtxDir(args.d, args.dir)
