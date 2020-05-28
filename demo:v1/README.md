# gesture-classifier 手势分析判别  
**BY** [GJ](https://github.com/Acemyzoe)

## 使用指南 
`python gesture_model.py`

主函数 : 

```python
#使用预训练模型
mymodel = CNN(False)
analysis(mymodel,'./imgs/ssstop1.png') #分析图片
capture(mymodel) #从摄像头获取手势进行分析

#训练网络
mymodel = CNN(True)
trainmodel(mymodel)
```

## 环境配置
  * ubuntu18.04 , anaconda-python3.7
  * 推荐使用Anaconda（一个提供包管理和环境管理的python版本）。  [官网下载](https://www.anaconda.com/distribution/)
  * 推荐修改镜像地址：
  
      >pip install pip -U 
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
  
* 安装需要的python库：(缺少相应的库可用conda或者pip自行安装) 
    > * opencv-python
    > * keras or tensorflow.keras
    > * tensorflow
