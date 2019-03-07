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

statis = {
    1: 0,
    2: 0,
    3: 0,
    11: 0,
    12: 0,
    13: 0,
    14: 0,
    4624: 0,
    4688: 0
}


def statis_all():
    hosts = {}
    with open('winlogbeat1224.json') as infile:
        for line in infile:
            line = json.loads(line)
            event_id = line['_source']['event_id']
            source_name = line['_source']['source_name']
            if event_id in event_type and event_type[event_id] == source_name:
                statis[event_id] += 1
                if 'host' not in line['_source']:
                    print(line)
                    continue
                host = line['_source']['host']['name']
                if host in hosts:
                    hosts[host] += 1
                else:
                    hosts[host] = 1
        print('Host'.ljust(40), 'Event Count'.ljust(4))
        for host in hosts:
            print(host.ljust(40), str(hosts[host]).ljust(40))

def statis_all_user():
    user = {}
    with open('winlogbeat5.json') as infile:
        for line in infile:
            line = json.loads(line)
            event_id = line['_source']['event_id']
            source_name = line['_source']['source_name']
            if event_id == 4624 and source_name == event_type[4624]:  
                if 'User' not in line['_source']['event_data']:
                    continue
                user = line['_source']['event_data']['User']

def statis_logon():
    user_logon = {}
    with open('winlogbeat5.json') as infile:
        for line in infile:
            event = json.loads(line)
            event_id = event['_source']['event_id']
            source_name = event['_source']['source_name']
            if event_id in event_type and source_name == event_type[event_id]:
                if event_id == 4688 \
                  and 'TargetDomainName' in event['_source']['event_data'] \
                  and 'TargetUserName' in event['_source']['event_data']:
                    target_user = event['_source']['event_data']['TargetDomainName'] \
                                + '/'  \
                                + event['_source']['event_data']['TargetUserName']
                    if target_user in user_logon:
                        user_logon[target_user] += 1
                    else:
                        user_logon[target_user] = 1
    for user in user_logon:
        print(user.ljust(40), str(user_logon[user]).ljust(40))

def statis_lyw_3():
    count = 0
    a = 0
    users = {}
    with open('winlogbeat5.json') as infile:
        outfile = open('ieid/lyw_event', 'w')
        for line in infile:
            event = json.loads(line)
            event_id = event['_source']['event_id']
            source_name = event['_source']['source_name']
            if event_id == 1 and source_name == event_type[1]:
                try:
                    user = event['_source']['event_data']['User']
                    # if user in users:
                    #     users[user] += 1
                    # else:
                    #     users[user] = 1
                    #     print(user, event['_source']['user']['name'])
                    if user == 'XXAQ\\baji':
                        print(event['_source']['event_data']['Image'])
                        count += 1
                except:
                    a += 1
                    # print(event['_source'])
    for user in users:
        print(user, users[user])

if __name__ == "__main__":
    statis_logon()