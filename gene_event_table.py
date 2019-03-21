#!/usr/bin/python
# short script for generate event table from the sick event-log-type.json file.
# output of the script is standard csv format
# usage: python gene_event_table.py

import sys
import csv
import json
from collections import OrderedDict

infile_name = 'intermediate/windows-event-log-type.json'
outfile_name = 'intermediate/events.csv'


def gene_csv(infile_name, outfile_name):
    ordered_fields = OrderedDict([
        ('name', None),
        ('event id', None),
        ('event source', None),
        ('operation', None),
        ('source', None),
        ('destination', None),
        ('description', None),
        ('note', None)
    ])


    outfile = open(outfile_name, 'w')
    csv_writer = csv.DictWriter(outfile, delimiter=';', fieldnames=ordered_fields)
    csv_writer.writeheader()
    with open(infile_name) as infile:
        for line in infile:
            if line.startswith('name'):
                i = line.index(':')
                assert(i > 0)
                row = {
                    'name': line[:i],
                    'description': line[i+1: -1]
                }
                csv_writer.writerow(row)
    outfile.close()


def extract_id(event_id):
    with open('winlogbeat1224.json') as infile:
        for line in infile:
            event = json.loads(line)
            event = event['_source']
            if event['event_id'] == event_id:
                source_name = event['source_name']
                outfile = open('ieid/%s.%d.json' % (source_name, event_id), 'w')
                json.dump(event, outfile, ensure_ascii=False)
                outfile.close()
                break

if __name__ == "__main__":
    if len(sys.argv) == 2:
        extract_id(int(sys.argv[1]))