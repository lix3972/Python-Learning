#### 从基础概念到实现，小白如何快速入门PyTorch  
https://blog.csdn.net/candy_gl/article/details/81201060  
转自：https://blog.csdn.net/Julialove102123/article/details/80487269  
PyTorch中文文档  

官网教材：https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html  

中文教材：chenyuntc/pytorch-book  https://mp.csdn.net/postedit/chenyuntc/pytorch-book (网页打不开)
https://morvanzhou.github.io/tutorials/machine-learning/torch/  教程

##### 第一步 github的 tutorials
   https://pytorch.org/tutorials/ （注意：要写上https://否则打不开网页）     尤其是那个60分钟的入门。    
  另外jcjohnson 的Simple examples to introduce PyTorch 也不错    
   imple examples to introduce PyTorch链接：https://github.com/jcjohnson/pytorch-examples
##### 第二步 example
  参考 pytorch/examples 实现一个最简单的例子(比如训练mnist )。 https://github.com/pytorch/examples  
####  第三步 通读doc PyTorch doc
  尤其是autograd的机制，和nn.module ,optim 等相关内容。文档现在已经很完善，而且绝大部分文档都是作者亲自写的，质量很高。  
  PyTorch doc链接： https://pytorch.org/docs/   
  autograd链接： https://pytorch.org/docs/notes/autograd.html  
  nn.module链接：https://pytorch.org/docs/nn.html%23torch.nn.Module  
#### 第四步 论坛讨论 PyTorch Forums
 PyTorch Forums链接： https://discuss.pytorch.org/    
 论坛很活跃，而且质量很高，pytorch的维护者(作者)回帖很及时的。每天刷一刷帖可以少走很多弯路，避开许多陷阱,消除很多思维惯性.尤其看看那些阅读量高的贴，刷帖能从作者那里学会如何写出bug-free clean and elegant 的代码。如果自己遇到问题可以先搜索一下，一般都能找到解决方案，找不到的话大胆提问，大家都很热心的。
#### 第五步 阅读源代码
   fork pytorch，pytorch-vision等。相比其他框架，pytorch代码量不大，而且抽象层次没有那么多，很容易读懂的。通过阅读代码可以了解函数和类的机制，此外它的很多函数,模型,模块的实现方法都如教科书般经典。还可以关注官方仓库的issue/pull request, 了解pytorch开发进展，以及避坑。  
还可以加入 slack群组讨论，e-mail订阅等  
总之 pytorch入门很简单，代码很优雅，是我用过的最Pythonic的框架. 欢迎入坑。  
注：Pythonic指：很python的python代码。类似：这很知乎(知乎特有的现象);这很百度(百度特有的行为)；这种表达。

参考网址的作者(陈云)写的教程：https://github.com/chenyuntc/pytorch-book 用notebook写的教程，里面还有很多有趣的例子，比如用GAN生成动漫头像，用CharRNN写唐诗，类Prisma的滤镜（风格迁移）和图像描述等 

### 安装 PyTorch 
例如以下选择在 Linux、Python 3.5 和 CUDA 9.1 的环境下安装 PyTorch：
conda install pytorch torchvision cuda91 -c pytorch

https://www.baidu.com/link%3Furl%3DZ0NtRMChanjECZY1UVxGh5lD6gJmKhDc18QvWDF4qqCbutWtgAaLVH0jfEyvwpvT

