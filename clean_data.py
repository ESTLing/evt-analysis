import json
import elasticsearch
from elasticsearch import helpers
from parse import event_type

def read_es(hostname):
    es = elasticsearch.Elasticsearch()
    return helpers.scan(es,
        query={
            'query': {
                'bool': {
                    'must': [
                        { 'term': {'host.name.keyword': hostname} }
                    ]
                }
            },
            'sort': [
                { '@timestamp': 'asc' }
            ]
        },
        preserve_order=True,
        index='winlogbeat')


def read_target_es():
    es = elasticsearch.Elasticsearch()
    return helpers.scan(es,
        query={
            'query': {
                'match_all': {}
            },
            'sort': [
                { '@timestamp': 'asc' }
            ]
        },
        preserve_order=True,
        index='winlogbeat_target')


def clean():
    hostname = 'DESKTOP-LTOOKJH'
    outfile = open('intermediate/' + hostname+'.json', 'w')
    last_event = None
    for record in read_es(hostname):
        event = record['_source']
        event.pop('user', None)
        if last_event == event:
            continue
        last_event = event
        json.dump(record, outfile, ensure_ascii=False)
        outfile.write('\n')
    outfile.close()


def test_time_order():
    last_time = ''
    es = elasticsearch.Elasticsearch()
    for event in helpers.scan(es,
        query={
            'query': {
                'match_all': {}
            },
            'sort': [
                { '@timestamp': 'asc' }
            ]
        },
        preserve_order=True,
        index='winlogbeat_target'):
        timestamp = event['_source']['@timestamp']
        if last_time > timestamp:
            print('Time order uncorrect')
        else:
            last_time = timestamp


def statis_id():
    statis = {}
    for event in read_target_es():
        event_id = event['_source']['event_id']
        source_name = event['_source']['source_name']
        if source_name not in statis:
            statis[source_name] = {}
        if event_id not in statis[source_name]:
            statis[source_name][event_id] = 1
        else:
            statis[source_name][event_id] += 1
    print('Event ID'.ljust(20), 'Event Count')
    for key in statis:
        print('\n' + key)
        for _id in statis[key]:
            print(str(_id).ljust(20), statis[key][_id])
        

def extract_actions():
    process_actions = []
    process_id = {}
    isolate_event = 0
    for event in read_target_es():
        event_id = event['_source']['event_id']
        source_name = event['_source']['source_name']
        if source_name == 'Microsoft-Windows-Sysmon' and event_id == 1:
            data = event['_source']['event_data']
            pid = data['ProcessId']
            process_actions.append({
                'timestamp': data['UtcTime'],
                'pid': pid,
                'image': data['Image'],
                'actions': []
            })
            process_id[pid] = process_actions[-1]
            ppid = data['ParentProcessId']
            if ppid not in process_id:
                isolate_event += 1
                continue
            process_id[ppid]['actions'].append('CreateProcess:'+data['Image'])
        elif source_name == 'Microsoft-Windows-Sysmon' and event_id == 3:
            data = event['_source']['event_data']
            pid = data['ProcessId']
            if pid not in process_id:
                isolate_event += 1
                continue
            process_id[pid]['actions'].append('Network:'+ \
                                            data['DestinationIp'] + ':' + \
                                            data['DestinationPort'])
        elif source_name == 'Microsoft-Windows-Sysmon' and event_id == 11:
            data = event['_source']['event_data']
            pid = data['ProcessId']
            if pid not in process_id:
                isolate_event += 1
                continue
            process_id[pid]['actions'].append('FileCreate:' + \
                                            data['TargetFilename'])
        elif source_name == 'Microsoft-Windows-Sysmon' and event_id == 12:
            data = event['_source']['event_data']
            pid = data['ProcessId']
            if pid not in process_id:
                isolate_event += 1
                continue
            process_id[pid]['actions'].append(data['EventType'] + ':' + \
                                            data['TargetObject'])
        elif source_name == 'Microsoft-Windows-Sysmon' and event_id == 13:
            data = event['_source']['event_data']
            pid = data['ProcessId']
            if pid not in process_id:
                isolate_event += 1
                continue
            process_id[pid]['actions'].append(data['EventType'] + ':' + \
                                            data['TargetObject'])
    print('Total processes count:', len(process_actions))
    print('isolate event count:', isolate_event)
    with open('intermediate/actions.json', 'w') as outfile:
        json.dump(process_actions, outfile, ensure_ascii=False)

if __name__ == "__main__":
    extract_actions()