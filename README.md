# IMPORT DATA
所有的数据应该存储在elasticsearch中，并从elasticsearch中读取。`DESKTOP-LTOOKJH.json`是
渗透实验的目标主机记录的日志，导入日志的命令为：
```shell
elasticdump --input=DESKTOP-LTOOKJH.json --output=http://127.0.0.1:9200/winlogbeat_target --type=data
```
elasticdump需要通过npm安装。

# TODO
- [ ] 收集事件的信息
- [X] 进程行为简单分析
- [ ] 正则表达式处理日志(暂时搁置)
- [ ] 使用[bar图](https://www.echartsjs.com/examples/editor.html?c=bar-y-category-stack)可视化目标主机事件类型