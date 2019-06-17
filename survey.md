# Windows event的理解
对event的理解的目的有两个：
1. 找到可能与用户行为有关的event id
   - 根据event的解释判断Windows产生这些event的情况
2. 寻找这些event的关键字段
   - 关键字段指可以无歧义的区分行为的字段，比如固定ip情况下的ip字段，server端的port号

## Logon/Logoff
Windows里Logon需要authentication，因此一次登录会产生两条事件，一条记录Account Logon(authenticate)，
另一条记录Logon(账户获得访问权限)。前者在域账户的登录中被记录在DC，本地登录中记录在本地，后者记录在被登录
的主机上[^1]。


## PS
- 以目前的数据而言，大部分的分析是无用的，比如网络登录在我们的数据采集中根本就没有。
- 目前采集数据里域账户基本上没有用户活动，都在本地账户完成日常事务。

# 周期性的分析
与用户行为挖掘的关联：行为周期性建模可以帮助检测异常或协助预测
经典的周期性分析步骤：平滑处理+加法(乘法)模型

## 周期检测算法
- Fourier transform
  - 难以处理稀疏数据
  - 需要均匀采样的数据
- auto-correlation
  - 难以找到unique period
  - 需要均匀采样的数据
- 稀疏数据周期检测算法[^2]

Windows event日志中包含这两类数据，比如用户登录数据是极其稀疏的，network connect数据是非稀疏的

## 超参数
- time window：针对稀疏事件，以不同window统计事件可能会观察到不一样的周期性
- upper & lower limit of period

## PS
- 日志事件的周期性不存在趋势
- windows日志中周期性的寻找不需要极为复杂的周期性检测算法，不需要考虑趋势

# 下一步
剔除了周期性的事件剩下什么

[^1]: https://www.forwarddefense.com/pdfs/Event_Log_Analyst_Reference.pdf
[^2]: Li, Wang, and Han, “Mining Event Periodicity from Incomplete Observations.”