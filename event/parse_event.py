def deal_event_1(event, host, userLog):
    data = event['event_data']
    username = data['User']
    if username in host['user']:
        host['user'][username]['processes'].append({
            'timestamp': data['UtcTime'],
            'pid': data['ProcessId'],
            'image': data['Image'],
            'actions': []
        })
        host['process'][data['ProcessId']] = username
        ParentProcessId = data['ParentProcessId']
        if ParentProcessId in host['process']:
            username = host['process'][ParentProcessId]
            if username in host['user']:
                for i in range(len(host['user'][username]['processes'])-1, -1, -1):
                    process = host['user'][username]['processes'][i]
                    if process['pid'] == ParentProcessId:
                        action_string = 'CreateProcess:' + \
                                        data['CommandLine']
                        process['actions'].append(action_string)
                        break
    else:
        print(data['User'])



def deal_event_2(event, host, userLog):
    pass


def deal_event_3(event, host, userLog):
    data = event['event_data']
    ProcessId = data['ProcessId']
    if ProcessId in host['process']:
        username = host['process'][ProcessId]
        if username in host['user']:
            for i in range(len(host['user'][username]['processes'])-1, -1, -1):
                process = host['user'][username]['processes'][i]
                if process['pid'] == ProcessId:
                    action_string = 'Network:' + \
                                    data['Image'] + ':' + \
                                    data['DestinationIp'] + ':' + \
                                    data['DestinationPort']
                    process['actions'].append(action_string)
                    break


def deal_event_11(event, host, userLog):
    data = event['event_data']
    ProcessId = data['ProcessId']
    if ProcessId in host['process']:
        username = host['process'][ProcessId]
        if username in host['user']:
            for i in range(len(host['user'][username]['processes'])-1, -1, -1):
                process = host['user'][username]['processes'][i]
                if process['pid'] == ProcessId:
                    action_string = 'FileCreate:' + \
                                    data['Image'] + ':' + \
                                    data['TargetFilename']
                    process['actions'].append(action_string)
                    break


def deal_event_12(event, host, userLog):
    data = event['event_data']
    ProcessId = data['ProcessId']
    if ProcessId in host['process']:
        username = host['process'][ProcessId]
        if username in host['user']:
            for i in range(len(host['user'][username]['processes'])-1, -1, -1):
                process = host['user'][username]['processes'][i]
                if process['pid'] == ProcessId:
                    action_string = data['EventType'] + ':' + \
                                    data['Image'] + ':' + \
                                    data['TargetObject']
                    process['actions'].append(action_string)
                    break


def deal_event_13(event, host, userLog):
    data = event['event_data']
    ProcessId = data['ProcessId']
    if ProcessId in host['process']:
        username = host['process'][ProcessId]
        if username in host['user']:
            for i in range(len(host['user'][username]['processes'])-1, -1, -1):
                process = host['user'][username]['processes'][i]
                if process['pid'] == ProcessId:
                    action_string = data['EventType'] + ':' + \
                                    data['Image'] + ':' + \
                                    data['TargetObject']
                    process['actions'].append(action_string)
                    break


def deal_event_4624(event, host, userLog):
    data = event['event_data']
    username = data['TargetDomainName'] + '\\' + data['TargetUserName']
    if username in host['user']:
        if username not in userLog:
            userLog[username] = []
        userLog[username].append(host['user'][username])
    host['user'] = {}
    host['process'] = {}
    host['user'][username] = {
        'logon': event['@timestamp'],
        'processes': []
    }


def deal_event_4688(event, host, userLog):
    pass
