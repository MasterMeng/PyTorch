# PyTorch 卷积模块  

PyTorch作为一个深度学习库，卷积神经网络是其中一个最为基础的模块，卷积神经网络中的所有的层结构都可以通过nn这个包调用。接下来介绍如何调用每种层结构以及每个函数中的参数。  

## 卷积层  

nn.Conv2d()就是PyTorch中的卷积模块，里面常用的参数有5个，分别是in_channels，out_channels，kernel_size，stride，padding，除此之外还有参数dilation，groups，bias。  

in_channels对应的是输入数据体的深度；out_channels表示输出数据体的深度；kernel_size表示滤波器（卷积核）的大小，可以使用一个数字来表示高和宽相同的卷积核，比如kernel_size = 3，也可以使用不同的数字来表示高和宽不同的卷积核，比如kernel_size = (3,2)；stride表示滑动的步长；padding = 0表示四周不进行零填充，而padding= 1表示四周进行1个像素的零填充；bias是一个布尔值，默认bias=True，表示使用偏置；groups表示输出数据体深度上和输入数据体深度上的联系，默认groups=1，也就是所有的输出和输入都是相关联的，如果groups=2，表示输入的深度被分割成两份，输出的深度也被分割成两份，它们之间分别对应起来，所以要求输出和输入都必须能被groups整除；dilation表示卷积对于输入数据体的空间间隔，默认dilation=1。  

## 池化层  

nn.MaxPool2d()表示网络中的最大值池化，其中的参数有kernel_size、stride、padding、dilation、return_indices、ceil_mode。  

* kernel_size、stride、padding、dilation与之前卷积层中介绍的含义相同；
* return_indices表示是否返回最大值所处的下标，默认 return_indices=False；
* ceil_mode 表示使用一些方格代替层结构，默认 ceil_mode=False，一般都不会设置这些参数。
  
nn.AvgPool2d()表示均值池化，里面的参数和nn.MaxPool2d()类似，但多了一个参数 count_include_pad，这个参数表示计算均值是是否包含零填充，默认count_include_pad=True。

一般使用较多的就是nn.MaxPool2d()和nn.AvgPool2d()，另外PyTorch还提供了一些别的池化层，如nn.LPPool2d()、nn.AdaptiveMaxPool2d()等，这些运用较少，可以查看官方的文档了解如何使用。  

## 提取层结构  

对于一个给定的模型，如果不想要模型中所有的层结构，只希望能够提取网络中的某一层或者几层，应该如何实现？  

nn.Module提供了几个重要的属性可以解决上述问题。第一个是children()，返回下一级模块的迭代器；modules()返回模型中所有模块的迭代器；named_children()及named_modules()，不仅返回模块的迭代器，还会返回网络层的名字。  

## 如何提取参数及自定义初始化  

有时候提取出层结构并不够，还需要对里面的参数进行初始化，那么如何提取出网络的参数并对其初始化呢？  

nn.Module里面也提供了关于参数的属性来解决上述问题：parameters()返回网络的全部参数的迭代器，named_parameters()不仅给出参数的迭代器，还给出网络层的名字。  

那么如何对权重做初始化呢？很简单，因为权重是一个Variable，所以只需要取出其中的data属性，然后对它进行所需要的处理就可以了。  