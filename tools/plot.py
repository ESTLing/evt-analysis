import plotly
from datetime import datetime, timedelta

target_file = 'intermediate/User11.log'

def dateRange(start, end, step=1):
    sec = (end - start).total_seconds() / 60
    return [start + timedelta(minutes=i) for i in range(0, int(sec), step)]


if __name__ == "__main__":
    time = dateRange(datetime(2012, 3, 21, 0),
           datetime(2012, 4, 13, 12), step=30)
    i = 0
    cnt = [0] * len(time)
    path = set()
    with open(target_file) as infile:
        for line in infile:
            line = line.split('|')
            timestamp = line[1] + ' ' + line[2][:8]
            timestamp = datetime.strptime(timestamp, '%d/%m/%Y %H:%M:%S')
            while timestamp > time[i]:
                cnt[i] = len(path)
                i+=1
            path.add(line[5])
    
    data = plotly.graph_objs.Scatter(
        x=time,
        y=cnt,
        mode = 'lines+markers',
        name='lines'
    )
    plotly.offline.plot([data], filename='intermediate/fileAccessPlot.html')