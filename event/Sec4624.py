import json
from datetime import datetime, timezone, timedelta
import elasticsearch
from elasticsearch import helpers
import plotly
import random

def read_es(_id):
    es = elasticsearch.Elasticsearch()
    return helpers.scan(es,
                        query={
                            'query': {
                                'bool': {
                                    'must': [
                                        {
                                            'term': {'eventId': _id}
                                        }
                                    ]
                                }
                            },
                            'sort': [
                                {'@timestamp': 'asc'}
                            ]
                        },
                        preserve_order=True,
                        index='syslog*')


def dateRange(start, end, step=1):
    sec = (end - start).total_seconds() / 60
    return [start + timedelta(minutes=i) for i in range(0, int(sec), step)]


if __name__ == "__main__":
    statis = {}
    time = dateRange(datetime(2019, 3, 13, 20),
                    datetime(2019, 3, 15, 23), step=1)
    i = 0
    for event in read_es(4624):
        event = event['_source']
        timestamp = event['@timestamp'].split('.')[0]
        timestamp = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S')
        timestamp = timestamp + timedelta(hours=8)
        while timestamp > time[i]:
            i += 1
        
        IpAddress = event['event_data']['IpAddress']
        if not IpAddress.startswith('192.168.1'): continue
        TargetUserName = event['event_data']['TargetUserName']
        if not TargetUserName == "WINCC$": continue
        LogonProcessName = event['event_data']['LogonProcessName']
        src = event['src']
        logonType = event['logonType']
        IPport = event['event_data']['IpPort']
        AuthenticationPackageName = event['event_data']['AuthenticationPackageName']
        key = IpAddress + ';' + src + ';' + TargetUserName + ';' + AuthenticationPackageName
        if key not in statis:
            statis[key] = [0] * len(time)
        statis[key][i] += 1
    
    data = []
    for key in statis:
        data.append(plotly.graph_objs.Scatter(
            x=time,
            y=statis[key],
            name=key,
            line=dict(
                color=("#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])),
                width=2
            )
        ))
    layout = dict(title='Event 4624 Statistics',
                  xaxis=dict(title='Time'),
                  yaxis=dict(title='Count'))
    plotly.offline.plot(dict(data=data, layout=layout), filename='Sec-4624-line.html')