# This script used for collect data from elasticsearch RESTful API
# It consists of the following function
# 1. collection event according to EVENT ID
# 2. show basic infomation of the specified condition
# [TODO] 3. show basic information of EVENT ID 

import sys
import argparse
import json
from datetime import datetime
try:
    from elasticsearch import Elasticsearch, helpers
except ImportError:
    print('Fail to import elasticsearch, please using the following command to install elasticseach python client:')
    print(' - pip install elasticsearch')
    exit(0)


ELASTICSEARCH_IP = '10.129.51.190'
ELASTICSEARCH_PORT = '9200'
ELASTICSEARCH_INDEX = 'winlogbeat-7.4.2-2019.11.02-000001'


def es_connect(ip, port):
    es = Elasticsearch(hosts=ip+':'+port)
    return es


def es_count(es, index, query):
    res = es.count(index=index, body=query)
    return res['count']


def es_scroll(es, index, query, cnt, prefix='', size=40):
    # res = es.search(index=index, body=query)
    def show(j):
        x = int(size*j/cnt)
        sys.stdout.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, cnt))
        sys.stdout.flush()        
    show(0)
    
    for i, event in enumerate(helpers.scan(es, query=query, scroll='5m', size=1000)):
        yield event
        show(i+1)
    sys.stdout.write('\n')
    sys.stdout.flush()


def build_query(event_id=None, hostname=None, start_time=None, end_time=None):
    query = {'query': {
        'bool': {
            'filter': [
                {'term': {'winlog.channel': 'Security'}}
                # {'term': {'winlog.provider_name': 'Microsoft-Windows-Security-Auditing'}}
            ]
        }
    }}
    if event_id is not None:
        query['query']['bool']['filter'].append({
            'terms': {
                'winlog.event_id': event_id
            }
        })
    if hostname is not None:
        query['query']['bool']['filter'].append({
            'terms': {
                'winlog.computer_name': hostname
            }
        })
    timerange = {
        'range': {
            '@timestamp': {
                'time_zone': '+08:00'
            }
        }
    }
    if start_time is not None:
        timerange['range']['@timestamp']['gte'] = start_time
    if end_time is not None:
        timerange['range']['@timestamp']['lte'] = end_time
    if start_time is not None or end_time is not None:
        query['query']['bool']['filter'].append(timerange)
    return query


def agg_hostname_info(es, index, query):
    query['aggs'] = {
        'hostname': {
            'terms': { 'field': 'winlog.computer_name' }
        }
    }
    res = es.search(index=index, body=query)
    del query['aggs']

    print(''.ljust(5), 'Availabel hostname'.ljust(20), 'event statistics')
    for host in res['aggregations']['hostname']['buckets']:
        print(''.ljust(5), host['key'].ljust(20), host['doc_count'])
    print('')


def parse_and_check_event_id(event_id):
    if event_id is None: return None
    id_list = event_id.split(',')
    for id_ in id_list:
        if not id_.isdigit():
            print('%s is not a valid event ID, please use correct format.(For multiple event ID, use "," to split them)' % id_)
            exit(0)
    return id_list


def parse_hostname(host_name):
    if host_name is None: return None
    return host_name.split(',')


def parse_and_check_time(timestr):
    if timestr is None: return None
    try:
        localtime = datetime.strptime(timestr, '%Y-%m-%d')
    except ValueError:
        print('The format of time string is incorrect, the only supported time format is "%Y-%m-%d"')
        exit(0)
    return localtime


def verify_info(event_id, host_name, start_time, end_time, cnt):
    print('Please verify the basic information before actually write into files.\n')
    print('(None means match all)')
    print('Event ID: ', event_id)
    print('Host Name: ', host_name)
    print('Time Range: ', start_time, ' - ', end_time, '\n')
    print('The number of events match all the conditions: ', cnt)
    if cnt > 10000:
        print('WARNING: You are trying to read a large number of events(greater than 10,000), which may cause the result file is huge, please be careful.')
    
    i = ''
    while i == '':
        i = input('Are you sure to continue? (Y/N)')
    return i.upper() == 'Y'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--event_id', help='filter event id, split by ","')
    parser.add_argument('--host_name', help='filter host, split by ","')
    parser.add_argument('--start_time', help='the time to start collect, in format of "1970-1-1"')
    parser.add_argument('--end_time', help='the time to stop collect, in format of "1970-1-1"')
    parser.add_argument('dst_file', help='the destination which save data')
    args = parser.parse_args()

    event_id = parse_and_check_event_id(args.event_id)
    host_name = parse_hostname(args.host_name)
    start_time = parse_and_check_time(args.start_time)
    end_time = parse_and_check_time(args.end_time)
    
    es = es_connect(ELASTICSEARCH_IP, ELASTICSEARCH_PORT)
    query = build_query(event_id, host_name, start_time, end_time)
    cnt = es_count(es, ELASTICSEARCH_INDEX, query)

    if host_name is None:
        agg_hostname_info(es, ELASTICSEARCH_INDEX, query)

    if verify_info(event_id, host_name, start_time, end_time, cnt):
        with open(args.dst_file, 'w') as f:
            for batch in es_scroll(es, ELASTICSEARCH_INDEX, query, cnt):
                json.dump(batch, f)
                f.write('\n')
        print('You are done! The data has been write into %s, but notice that these events are unordered, remember it before using the data.' % args.dst_file)
    else:
        print('Abort by user.')


if __name__ == "__main__":
    main()