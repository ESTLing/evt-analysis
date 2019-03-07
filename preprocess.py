# 从原始数据中提取目标主机的日志

import json
from datetime import datetime, timedelta, timezone
import elasticsearch
from elasticsearch import helpers

from event.parse_event import *
from event.event_type import event_dict


hostList = {}
userLog = {}


def read_all_event():
    with open('../1224exp/winlogbeat1224.json') as infile:
        for line in infile:
            event = json.loads(line)['_source']
            event_id = event['event_id']
            source_name = event['source_name']
            if event_id in event_dict and event_dict[event_id] == source_name:
                yield event, event_id


def preprocess_data():
    for event, event_id in read_all_event():
        func = 'deal_event_'+str(event_id)
        hostname = event['host']['name']
        if hostname not in hostList:
            hostList[hostname] = {
                'user': {},
                'process': {}
            }
        eval(func)(event, hostList[hostname], userLog)
    with open('rst.json', 'w') as outfile:
        json.dump(userLog, outfile)


def read_es(start_time, end_time):
    es = elasticsearch.Elasticsearch()
    return helpers.scan(es,
        query={
            'query': {
                'bool': {
                    'must': [
                        {
                            'range': {
                                '@timestamp': {
                                    'gte': start_time,
                                    'lte': end_time
                                }
                            }
                        }
                    ]
                }
            },
            'sort': [
                { '@timestamp': 'asc' }
            ]
        },
        preserve_order=True,
        index='winlogbeat')


def statis_host():
    host_dict = {}
    start_time = datetime(2018, 12, 13, 20, 52)
    start_time = datetime.utcfromtimestamp(start_time.timestamp())
    end_time = datetime(2018, 12, 13, 21, 5)
    end_time = datetime.utcfromtimestamp(end_time.timestamp())
    print(end_time)
    for event in read_es(start_time, end_time):
        event = event['_source']
        if 'host' not in event:
            print('No host field')
        hostname = event['host']['name']
        if hostname in host_dict:
            host_dict[hostname] += 1
        else:
            host_dict[hostname] = 1
    for host in host_dict:
        print(host.ljust(40), host_dict[host])


def incise_data(start_time, end_time):
    utc_start_time = datetime.utcfromtimestamp(start_time.timestamp())
    utc_end_time = datetime.utcfromtimestamp(end_time.timestamp())
    event_list = []
    last_event = {}
    for event in read_es(utc_start_time, utc_end_time):
        event = event['_source']
        if 'host' not in event:
            print('No host field')
        # elif event['host']['name'] != 'WIN-CU7KT80AT3C':
        elif event['host']['name'] != 'DESKTOP-LTOOKJH':
            # print('log from host: %s' % event['host']['name'])
            continue
        else:
            if last_event \
                and last_event['@timestamp'] == event['@timestamp']\
                and last_event['event_id'] == event['event_id']\
                and last_event['event_data'] == event['event_data']:
                print(last_event['@timestamp'])
                continue
            last_event = event
            event_list.append(event)
    print(len(event_list))
    with open(start_time.strftime('%m%d%H%M') + '-'
        + end_time.strftime('%m%d%H%M') + '.json', 'w') as o:
        json.dump(event_list, o, ensure_ascii=False)


if __name__ == "__main__":
    # statis_host()
    incise_data(datetime(2018, 12, 12, 20, 35), datetime(2018, 12, 12, 20, 40))