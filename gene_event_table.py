#!/usr/bin/python
# short script for generate event table from the sick event-log-type.json file.
# output of the script is standard markdown table format
# usage: python gene_event_table.py <path-to-event-type-file>

import sys

infile_name = 'intermediate/windows-event-log-type.json'
if len(sys.argv) == 2:
    infile_name = sys.argv[1]

print('| name | event id | event source | operation | source | destination | description | note |')
print('| ---- | :------: | :----------: | :-------: | :----: | :---------: | :---------- | :--- |')

with open(infile_name) as infile:
    for line in infile:
        if line.startswith('name'):
            i = line.index(':')
            assert(i > 0)
            print('|', line[:i], '| | | | | |', line[i+1:-1], '| |')
