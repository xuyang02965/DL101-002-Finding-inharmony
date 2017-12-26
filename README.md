# DL101-002-Finding-inharmony

## 项目背景

* 利用掌握的深度学习技术开发一个有趣的小项目。一方面巩固课程知识；另一方面实际演练相关技术的工程实践。

* 小游戏「找别扭」—— 找出一副图片中的不协调物体，例如在一片荒芜沙漠中的红玫瑰。

## 项目涉及的技术要点：

* 图片中不同物体的标注。

* 判断图片中物体之间的合理关联，找出不符合当前上下文的物体。

## 项目的技术难点和折衷目标：

想实现通用的寻找不一致物体的功能非常难，因为：

* 没有足够的数据使模型认识各种物体。

* 没有足够的数据使模型识别物体间合理的关联。

所以，**本次项目只实现识别特定不协调物体的功能**。

## 可用数据集

从kaggle平台上找的图像数据集：
* [从卫星拍摄的地表图像](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data)
* [自然保护鱼类监控图像](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/data)
* [Boston 街景](http://cbcl.mit.edu/software-datasets/streetscenes/)
* [Labelme](http://www.ais.uni-bonn.de/download/datasets.html)

用以上数据集，可以将鱼类图像叠加到街景图像上去，将鱼作为异常物体。这样我们的游戏就是**找出街景图像中的鱼**。
  
