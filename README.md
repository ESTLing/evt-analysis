# IMPORT DATA
所有的数据应该存储在elasticsearch中，并从elasticsearch中读取。`DESKTOP-LTOOKJH.json`是
渗透实验的目标主机记录的日志，导入日志的命令为：
```shell
elasticdump --input=DESKTOP-LTOOKJH.json --output=http://127.0.0.1:9200/winlogbeat_target --type=data
```
elasticdump需要通过npm安装。

