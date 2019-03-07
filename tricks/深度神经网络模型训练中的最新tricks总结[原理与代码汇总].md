# 深度神经网络模型训练中最新tricks总结[原理与代码汇总]
<br><br>
计算机视觉主要问题有图像分类、目标检测和图像分割等。针对图像分类任务，提升准确率的方法路线有两条，一个是模型的修改，另一个是各种数据处理和训练的技巧(tricks)。图像分类中的各种技巧对于目标检测、图像分割等任务也有很好的作用，因此值得好好总结。本文在精读论文的基础上，总结了图像分类任务的各种tricks如下：

- Warmup
- Linear scaling learning rate
- Label-smoothing
- Random image cropping and patching
- Knowledge Distillation
- Cutout
- Random erasing
- Cosine learning rate decay
- Mixup training
- AdaBoud
- AutoAugment
- 其他经典的tricks

## Warmup
学习率是神经网络训练中最重要的超参数之一，针对学习率的技巧有很多。Warm up是在ResNet论文[1]中提到的一种学习率预热的方法。由于刚开始训练时模型的权重(weights)是随机初始化的(全部置为0是一个坑，原因见[2])，此时选择一个较大的学习率，可能会带来模型的不稳定。学习率预热就是在刚开始训练的时候先使用一个较小的学习率，训练一些epoches或iterations，等模型稳定时再修改为预先设置的学习率进行训练。论文[1]中使用一个110层的ResNet在cifar10上训练时，先用0.01的学习率训练直到训练误差低于80%(大概训练了400个iterations)，然后使用0.1的学习率进行训练。

上述的方法是constant warmup，18年Facebook又针对上面的warmup进行了改进[3]，因为从一个很小的学习率一下变为比较大的学习率可能会导致训练误差突然增大。论文[3]提出了gradual warmup来解决这个问题，即从最开始的小学习率开始，每个iteration增大一点，直到最初设置的比较大的学习率。

Gradual warmup代码如下：
```
from torch.optim.lr_scheduler import _LRScheduler

class GradualWarmupScheduler(_LRScheduler):
    
""" 
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    
def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
    self.multiplier = multiplier
    if self.multiplier <= 1.:
        raise ValueError('multiplier should be greater than 1.')
    self.total_epoch = total_epoch
    self.after_scheduler = after_scheduler
    self.finished = False
    super().__init__(optimizer)
        
def get_lr(self):
    if self.last_epoch > self.total_epoch:
         if self.after_scheduler:
             if not self.finished:
                 self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                 self.finished = True
             return self.after_scheduler.get_lr()
         return [base_lr * self.multiplier for base_lr in self.base_lrs]
    return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
    
def step(self, epoch=None):
    if self.finished and self.after_scheduler:
       return self.after_scheduler.step(epoch)
    else:
       return super(GradualWarmupScheduler, self).step(epoch)
```
## Linear scaling learning rate
Linear scaling learning rate是在论文[3]中针对比较大的batch size而提出的一种方法。

在凸优化问题中，随着批量的增加，收敛速度会降低，神经网络也有类似的实证结果。随着batch size的增大，处理相同数据量的速度会越来越快，但是达到相同精度所需要的epoch数量越来越多。也就是说，使用相同的epoch时，大batch size训练的模型与小batch size训练的模型相比，验证准确率会减小。

上面提到的gradual warmup是解决此问题的方法之一。另外，linear scaling learning rate也是一种有效的方法。在mini-batch SGD训练时，梯度下降的值是随机的，因为每一个batch的数据是随机选择的。增大batch size不会改变梯度的期望，但是会降低它的方差。也就是说，大batch size会降低梯度中的噪声，所以我们可以增大学习率来加快收敛。

具体做法很简单，比如ResNet原论文[1]中，batch size为256时选择的学习率是0.1，当我们把batch size变为一个较大的数b时，学习率应该变为 0.1 × b/256。
## Label-smoothing
在分类问题中，我们的最后一层一般是全连接层，然后对应标签的one-hot编码，即把对应类别的值编码为1，其他为0。这种编码方式和通过降低交叉熵损失来调整参数的方式结合起来，会有一些问题。这种方式会鼓励模型对不同类别的输出分数差异非常大，或者说，模型过分相信它的判断。但是，对于一个由多人标注的数据集，不同人标注的准则可能不同，每个人的标注也可能会有一些错误。模型对标签的过分相信会导致过拟合。

标签平滑(Label-smoothing regularization,LSR)是应对该问题的有效方法之一，它的具体思想是降低我们对于标签的信任，例如我们可以将损失的目标值从1稍微降到0.9，或者将从0稍微升到0.1。标签平滑最早在inception-v2[4]中被提出，它将真实的概率改造为：
</br>
 <img src="https://github.com/NKvision428/share-code/blob/master/tricks/imgs/1.webp"  div align=center />
 其中，ε是一个小的常数，K是类别的数目，y是图片的真正的标签，i代表第i个类别，q_i是图片为第i类的概率。

总的来说，LSR是一种通过在标签y中加入噪声，实现对模型约束，降低模型过拟合程度的一种正则化方法。

LSR代码如下：
```
import torch
import torch.nn as nn
class LSR(nn.Module):
    def __init__(self, e=0.1, reduction='mean'):
        super().__init__()
        self.log_softmax = nn.
        LogSoftmax(dim=1)
        self.e = e
        self.reduction = reduction
    def _one_hot(self, labels, classes, value=1):
        """
            Convert labels to one hot vectors
        Args:
            labels: torch tensor in format [label1, label2, label3, ...]
            classes: int, number of classes
            value: label value in one hot vector, default to 1
            Returns: 
                return one hot format labels in shape [batchsize, classes]
        """
        one_hot = torch.zeros(labels.size(0), classes)
        #labels and value_added  size must match
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)
        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)
        one_hot.scatter_add_(1, labels, value_added)
        
        return one_hot
    
def _smooth_label(self, target, length, smooth_factor):
    """convert targets to one-hot format, and smooth them.
    Args:
        target: target in form with [label1, label2, label_batchsize]

            length: length of one-hot format(number of classes)

            smooth_factor: smooth factor for label smooth



        Returns:

            smoothed labels in one hot format

        """

        one_hot = self._one_hot(target, length, value=
1
 - smooth_factor)

        one_hot += smooth_factor / length



        
return
 one_hot.to(target.device)
```

