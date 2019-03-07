# 在提取进程event list的基础上，按照session组织事件

import json

event_type = {
    1: 'Microsoft-Windows-Sysmon',
    2: 'Microsoft-Windows-Sysmon',
    3: 'Microsoft-Windows-Sysmon',
    11: 'Microsoft-Windows-Sysmon',
    12: 'Microsoft-Windows-Sysmon',
    13: 'Microsoft-Windows-Sysmon',
    14: 'Microsoft-Windows-Sysmon',
    4624: 'Microsoft-Windows-Security-Auditing',
    4688: 'Microsoft-Windows-Security-Auditing',
}

def get_session():
    with open('winlogbeat5.json') as infile:
        for line in infile:
            event = json.loads(line)
            event_id = event['_source']['event_id']
            source_name = event['_source']['source_name']
            if event_id == 4624 and source_name == event_type[4624]:
                if 'User' not in line['_source']['event_data']:
                    continue
                user = line['_source']['event_data']['User']


if __name__ == "__main__":
    pass