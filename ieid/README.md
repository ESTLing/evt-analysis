# 事件统计说明
## 字段说明
- name: 原json文件中的字段
- event id: 事件ID
- event source: [事件源](###事件源)
- operation: 事件操作的[类型](###操作类型)
- source: 产生行为的[主体](###主体)
- destination: 作为行为对象的[客体](###客体)
- description: 原json文件中的字段，用于查找事件的唯一可靠依据
- note: [额外说明](###额外说明)

### 事件源
Microsoft-Windows-Security-Auditing是比较特殊的一种事件源，其事件可以划分为两级的sub-category

### 操作类型
某些事件代表一类行为，暂且定义操作类型是用来区分不同行为的。

例1：Microsoft-Windows-Security-Auditing里的4624事件是登录事件，登录事件包含交互登录、网络登录、Service登录等类型，这一信息记录在Logon Information->Logon Type字段，它就是区分同一事件不同行为的操作类型。从安全意义来讲，区分不同类型的登录是有意义的。

例2：Microsoft-Windows-Security-Auditing里的5447事件是授权事件，被授予给某个账户的Privilege被记录在Privileges字段。同上面一样，以细粒度划分的话，不同的Privilege代表的含义是不同的，所以这里取该字段作为划分不同行为的标准。

如果事件本身代表的行为足够具体，该列可以为空。

### 主体
行为的发起者，包括host信息、user信息、process信息，越具体越好。

### 客体
行为的承受者，根据不同事件变化很大。常见的包括文件、账户、进程，或者没有。

### 额外说明
包括但不限于①特殊字段的说明②包含有效信息的其他字段的补充③与相关事件的关联

# 查找方式
Google