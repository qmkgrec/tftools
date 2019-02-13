# tftools
## 迭代记录
2019.02.13 增加embedding从其他模型初始化的功能

## 代码结构

--FCGen          数据读取模块\<br>  
----FCGen.py       数据读取接口函数\<br>  
----Generator.py   提供各种数据类型的处理（例如ID特征hash分桶、embeding等）\<br>  
--Trainer        模型训练模块\<br>  
----Evaluation.py  模型评估方法封装\<br>  
----InputFn.py     模型输入方法封装\<br>  
----Training.py    模型训练方法封装\<br>  

## 使用原理
特征的配置和特征间的组合在底层委托给[tf.feature_column](https://www.tensorflow.org/api_docs/python/tf/feature_column)来完成，这个模块用起来非常方便，和tfrecords格式的训练样本也很好配合。
参数配置的字段基本对应于不同类型feature_column的参数，以下是样例中的特征配置文件`Projects/Example/feature_spec.ini`

```ini

; date: 2018/9/21
; last edit: kevinshuang
; description: test ini file

[label]
ftype = numeric
shape = 1
default = 0.0

[wt]
ftype = numeric
shape = 1
default = 1.0
group = weight

;ugc info
;[score]
;ftype = bucketized
;shape = 1
;default = 0.0
;boundaries = 0.0,1.0
;auto_boundaries = 20
;group = linear
;
;[beat_percent]
;ftype = bucketized
;shape = 1
;default = 0.0
;boundaries = 0.0
;auto_boundaries = 20
;group = linear
;
;[scorerank]
;ftype = bucketized
;shape = 1
;default = 0.0
;boundaries = 0.0
;auto_boundaries = 20
;group = linear

[comment_num]
ftype = bucketized
shape = 1
default = 0.0
boundaries = 0.0,1.0,3.0,5.0,7.0,9.0,12.0,15.0,18.0,21.0,25.0,29.0,33.0,38.0,44.0,49.0,55.0,61.0,68.0,75.0,81.0,90.0,100.0,108.0,119.0,132.0,145.0,159.0,172.0,191.0,203.0,223.0,244.0,262.0,283.0,306.0,343.0,381.0,414.0,457.0,518.0,566.0,607.0,681.0,861.0,1008.0,1183.0,1464.0,1944.0
auto_boundaries = 50
group = linear

[comment_num_dnn]
fname = comment_num
ftype = numeric
shape = 1
default = 0.0
group = dnn,norm
min = 0.0
max = 3272.0
mean = 330.14227
std = 632.7193

[play_num]
ftype = bucketized
shape = 1
default = 0.0
boundaries = 0.0,97.0,814.9399999999996,2287.6000000000004,4148.0,6135.0,8252.0,11161.36,14110.0,17722.0,22023.0,27327.0,32434.06000000003,38050.0,45136.0,53623.0,64346.0,75385.0,88492.0,103178.0,122416.0,147524.1599999994,175338.0,203902.0,243814.0,295110.0,339066.0,377445.0,432909.0,484116.0,550727.0,603682.0800000001,677158.0,729377.0,819582.0,899400.0,972323.0,1033454.0,1124825.0,1206426.0,1329910.0,1459957.4800000065,1558994.0,1691467.0,1853402.0,2082594.0,2395177.0,2854963.0,3650181.0
auto_boundaries = 50
group = linear

```

#### 模型参数配置

把常用模型的训练脚本做成读取配置文件来设置参数，以下为样例中的线性模型配置文件`Projects/Example/conf.ini`

```ini
; input and model configurations

[input]
; 特征配置文件路径
spec = ./feature_spec.ini        

; 训练集和验证集路径，通常应该为不同的路径
train = ../../Data
dev = ../../Data

[train]
; random seed for tensorflow
seed = 19900816
batch_size = 64
epochs = 1
max_step = 100 

; 每隔多少batch保存一次checkpoint
steps_per_save = 20

; checkpoint目录，如果不填则为以下默认值 
;checkpoint = ./checkpoint/ckpt

; 验证集上效果最好的epoch的checkpoint目录，如果不填则为以下默认值 
;best_checkpoint = ./best_checkpoint/ckpt

; 从某个checkpoint目录恢复开始增量训练
;restore_from = ./checkpoint/ckpt


[model]
; number of output units for linear model
;units = 1
;combiner = sum
learning_rate = 0.1
l1_reg = 0.1
l2_reg = 0.1

; 一个epoch的最大迭代次数
; max_step = 10000
```

#### 运行样例
有一个运行样例在`Projects/Example`下
```
# 先激活virtualenv
[root@Tencent-SNG /data1/timmyqiu/qmkg/TensorFlowRec/LearningTools]# source /data1/timmyqiu/tfenv/bin/activate
# 在自己的工程目录下编写配置文件，运行自己的训练脚本train.py
(tfenv) [root@Tencent-SNG /data1/timmyqiu/qmkg/TensorFlowRec/LearningTools]# 
(tfenv) [root@Tencent-SNG /data1/timmyqiu/qmkg/TensorFlowRec/LearningTools]# cd Projects/Example/
(tfenv) [root@Tencent-SNG /data1/timmyqiu/qmkg/TensorFlowRec/LearningTools/Projects/Example]# ls
conf.ini  feature_spec.ini  test.py  train.py
(tfenv) [root@Tencent-SNG /data1/timmyqiu/qmkg/TensorFlowRec/LearningTools/Projects/Example]# python train.py --conf conf.ini 
2018-09-12 10:54:48,666 train directory: ../../Data
2018-09-12 10:54:48,666 train files: ['../../Data/part-r-00000']
2018-09-12 10:54:48,667 dev directory: ../../Data
2018-09-12 10:54:48,667 dev files: ['../../Data/part-r-00000']
2018-09-12 10:54:48,668 Creating iterators...
2018-09-12 10:54:48,710 Creating models...
2018-09-12 10:54:49,489 Start training for 1 epoch(s)
2018-09-12 10:54:50,144 Epoch 1/1
2018-09-12 10:54:50,557 - step 1, Train metrics: loss: 0.693
2018-09-12 10:54:50,559 - step 2, Train metrics: loss: 0.684
2018-09-12 10:54:50,561 - step 3, Train metrics: loss: 0.672
2018-09-12 10:54:50,563 - step 4, Train metrics: loss: 0.659
...
```


