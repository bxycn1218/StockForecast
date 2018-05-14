###LSTM股票预测
用LSTM算法进行股票预测，对比算法为BP

#### 文件结构

```
.
├── __pycache__
│   ├── bp.cpython-36.pyc
│   └── lstm.cpython-36.pyc
├── bp.py								bp实现文件
├── dataset
│   └── dataset_1.csv					数据集
├── lstm.py								lstm实现文件
├── manage.py							主程序
├── model_bp							bp训练模型保存位置
│   ├── checkpoint
│   ├── model.ckpt.data-00000-of-00001
│   ├── model.ckpt.index
│   └── model.ckpt.meta
└── model_lstm							lstm训练模型保存位置
    ├── checkpoint
    ├── modle.ckpt.data-00000-of-00001
    ├── modle.ckpt.index
    └── modle.ckpt.meta
```

#### 运行方法

需安装numpy pandas等扩展包

```
python manage.py
```

####输出结果

默认输出为模型迭代50次后的股票预测曲线图，如下图

![video](http://oygov02sc.bkt.clouddn.com/prediction.png)

修改`choice=0`输出为ame和acc随着迭代次数的变化图像

ame和acc为衡量指标，计算方式如下；
$$
MAE = \frac{1}{n_{samples}}\sum_{i=1}^{n_{samples-1}}|y_i-y_i^{\prime}|
$$

$$
ACC = \frac{1}{n_{samples}}\sum_{i=1}^{n_{samples-1}}|y_i-y_i^{\prime}|/y_i
$$

