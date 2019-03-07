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
---------
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
