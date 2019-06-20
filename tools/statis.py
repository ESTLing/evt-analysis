# This script will be used to statistic different aspect of host event

import json
import argparse


def loadEvent(file):
    ''' return event generator
    '''
    with open(file) as infile:
        for event in infile:
            event = json.loads(event)
            yield event


def filterEvent(iter_event, eventID, source=None):
    ''' Filter certain event ID
    '''
    for event in iter_event:
        if(event['Event']['System']['EventID'] != eventID):
            continue
        if(source != None and 
           event['Event']['System']['Provider']['Name'] != source):
            continue
        yield event


def filterField(iter_event, field, vlist):
    ''' Filter certain field which value in the list
    '''
    for event in iter_event:
        event_data = event['Event']['EventData']
        if(field not in event_data):
            continue
        if(event_data[field] in vlist):
            yield event


def listEvent(iter_event, field=None):
    for event in iter_event:
        event_data = event['Event']['EventData']
        if(field is None):
            print(event)
            continue
        s = []
        for f in field:
            if(f in event_data):
                s.append(event_data[f])
        print(':'.join(s))


def statisCount(iter_event, field):
    ''' Do the basic statistics
    '''
    cnt = {}
    cnt['total'] = 0
    for event in iter_event:
        cnt['total'] += 1
        event_data = event['Event']['EventData']
        if field not in event_data:
            cnt['Missing'] = cnt.get('Missing', 0) + 1
        else:
            cnt[event_data[field]] = cnt.get(event_data[field], 0) + 1
    print("Field Statistics for " + field)
    print("Total event count: %d" % cnt['total'])
    for key in sorted(cnt, key=cnt.get, reverse=True):
        print(key.ljust(40), cnt[key])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dst_file', help='the event file which you want to analysis')
    parser.add_argument('-fid', help='filter through event ID')
    parser.add_argument('-ffield', help='filter through filed')
    parser.add_argument('-c', help='category and count based on field')
    parser.add_argument('-field', help='field to analysis, print to output')
    args = parser.parse_args()

    iter_event = loadEvent(args.dst_file)
    if(args.fid):
        iter_event = filterEvent(iter_event, args.fid)
    if(args.ffield):
        ffield = args.ffield.split(':')
        iter_event = filterField(iter_event, ffield[0], ffield[1].split(','))
    if(args.c):
        statisCount(iter_event, args.c)
    elif(args.field):
        listEvent(iter_event, field=args.field.split(','))