# tftools

## 代码结构

----FCGen          数据读取模块
  --FCGen.py       数据读取接口函数
  --Generator.py   提供各种数据类型的处理（例如ID特征hash分桶、embeding等）
----Trainer        模型训练模块
  --Evaluation.py  模型评估方法封装
  --InputFn.py     模型输入方法封装
  --Training.py    模型训练方法封装
