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
                     datetime(2019, 3, 15, 10), step=10)
    i = 0
    for event in read_es(10):
        event = event['_source']
        if event['source_name'] != 'Microsoft-Windows-Sysmon': continue
        timestamp = event['@timestamp'].split('.')[0]
        timestamp = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S')
        timestamp = timestamp + timedelta(hours=8)
        while timestamp > time[i]:
            i += 1
        
        SourceImage = event['event_data']['SourceImage']
        TargetImage = event['event_data']['TargetImage']
        key = SourceImage + ';' + TargetImage
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
    layout = dict(title='Event 10(sysmon) Statistics',
                  xaxis=dict(title='Time'),
                  yaxis=dict(title='Count'))
    plotly.offline.plot(dict(data=data, layout=layout), filename='Sysmon-10-line.html')