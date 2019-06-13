# PyTorch基础  

## Tensor（张量）  

PyTorch里面处理的最基本的操作对象就是Tensor，表示的是一个多维矩阵，比如零维矩阵就是一个点，一维就是向量，二维就是一般的矩阵，多维就相当于一个多维数组，这和numpy是对应，而且PyTorch的Tensor可以和numpy的ndarray相互转换，唯一不同的是PyTorch可以在GPU上运行，而numpy的ndarray只能在CPU上运行。  

常用的不同数据类型的Tensor有**32位浮点型torch.FloatTensor**、**64位浮点型torch.DoubleTensor**、**16位整型torch.ShortTensor**、**32位整型torch.IntTensor**和**64位浮点型torch.LongTensor**。torch.Tensor默认的是*torch.FloatTensor*数据类型。  

## Variable（变量）  

Variable，变量，这个概念在numpy中是没有，是神经网络计算图里特有的一个概念，就是Variable提供了自动求导的功能。  

Variable和Tensor本质上没有区别，不过Variable会被放入一个计算图中，然后进行前向传播、反向传播、自动求导。  

首先Variable是在torch.autograd.Variable中，要将tensor变成Variable也非常简单，比如想让一个tensor a变成Variable，只需Variable(a)即可。  

Variable有三个比较重要的组成属性：**data，grad和grad_fn**。通过*data*可以取出Variable里面的tensor数值；*grad_fn*表示的是得到这个Variable的操作，比如通过加减还是乘除得到的；*grad*是这个Variable的反向传播梯度。  

## Dataset（数据集）  

在处理任何机器学习问题之前都需要数据读取，并进行预处理。PyTorch提供了很多工具使得数据的读取和预处理变得很容易。  

**torch.utils.data.Dataset**是代表这一数据的抽象类。你可以自己定义你的数据类，继承和重写这个抽象类，非常简单，只需要定义__len__和__getitem__这个两个函数：  
```python
class myDataset(Dataset):
    def __init__(self,csv_file,txt_file,root_dir,other_file):
        self.csv_data = pd.read_csv(csv_file)
        with open(txt_file,'r') as f:
            data_list = f.readlines()
        self.txt_data = data_list
        self.root_dir = root_dir

    def __len__(self):
        return len(self.csv_data)

    def __gettime__(self,idx):
        data = (self.csv_data[idx],self.txt_data[idx])
        return data
```  

通过上面的方式，可以定义我们需要的数据类，可以同迭代的方式来获取每一个数据，但这样很难实现缺batch，shuffle或者是多线程去读取数据，所以PyTorch中提供了一个简单的办法来做这个事情，通过torch.utils.data.DataLoader来定义一个新的迭代器，如下：  
```python
dataiter = DataLoader(myDataset,batch_size=32,shuffle=True,collate_fn=defaulf_collate)
```  

其中的参数都很清楚，只有collate_fn是标识如何去样本的，我们可以定义自己的函数来准确地实现想要的功能，默认的函数在一般情况下都是可以使用的。  

## nn.Module（模组）  

在PyTorch里面编写神经网络，所有的层结构和损失函数都来自于torch.nn，所有的模型构建都是从这个基类**nn.Module**继承的，于是有了下面的这个模板。  
```python
class net_name(nn.Module):
    def __init__(self,other_arguments):
        super(net_name,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size)
        # other network layer

    def forward(self,x):
        x = self.conv1(x)
        return x
```  

这样就建立一个计算图，并且这个结构可以复用多次，每次调用就相当于用该计算图定义的相同参数做一次前向传播，得益于PyTorch的自动求导功能，所以我们不需要自己编写反向传播。  

定义完模型之后，我们需要通过nn这个包来定义损失函数。常见的损失函数都已经定义在了nn中，比如均方误差、多分类的交叉熵以及二分类的交叉熵等等，调用这些已经定义好的的损失函数也很简单：  
```python
criterion = nn.CrossEntropyLoss()]
loss = criterion(output,target)
```  

这样就能求得我们的输出和真实目标之间的损失函数了。  

## torch.optim（优化）  

在机器学习或者深度学习中，我们需要通过修改参数使得损失函数最小化（或最大化），优化算法就是一种调整模型参数更新的策略。  

优化算法分为两大类：  

1. **一阶优化算法**

这种算法使用各个参数的梯度值来更新参数，最常用的一阶优化算法是梯度下降。所谓的梯度就是导数的多变量表达式，函数的梯度形成了一个向量场，同时也是一个方向，这个方向上方向导数最大，且等于梯度。梯度下降的功能是通过寻找最小值，控制方差，更新模型参数，最终使模型收敛，网络的参数更新公式如下：  

$\theta = \theta - \eta \times\frac{\sigma J(\theta)}{\sigma_\theta}$  

其中$\eta$是学习率，$\frac{\sigma J(\theta)}{\sigma_\theta}$是函数的梯度。这是深度学习里最常用的优化方法。

2. **二阶优化算法**  

二阶优化算法是用来二阶导数（也叫做Hessian方法）来最小化或最大化损失函数，主要基于牛顿法，但由于二阶导数的计算成本很高，所以这种方法并没有广泛使用。torch.optim是一个实现各种优化算法的包，大多数常见的算法都能到直接通过这个包来调用，比如随机梯度下降，以及添加动量的随机梯度下降，自适应学习率等。在调用的时候需要优话传入的参数，这些参数都必须是Variable，然后传入一些基本的设定，比如学习率和动量等。  

## 模型的保存和加载  

在PyTorch中使用torch.save来保存模型的结构和参数，有两种保存方式：  

1. 保存整个模型的结构信息和参数信息，保存对象是模型model；  
2. 保存模型的参数，保存的对象是模型的状态model.state_dict()。

可以按如下方式保存：save的第一个参数是保存对象，第二个参数是保存路径及名称：  
```python
    torch.save(model,'./model.pth')
    torch.save(model.state_dict(),'./model_state.pth')
```  

加载模型有两种对应于保存模型的方式：  

1. 加载完整的模型结构和参数信息，使用 load_model = torch.load('model.pth')，在网络较大的时候加载的时间较长，同时存储空间也比较大；  
2. 加载模型参数信息，需要先导入模型的结构，然后通过 model.load_state_dic(torch.load('model_state.pth'))来导入。


