###### PyTorch官网: https://pytorch.org/tutorials/  
###### 官方中文文档 https://pytorch.apachecn.org/  
###### PyTorch中文手册：https://github.com/zergtant/pytorch-handbook  
###### PyTorch常用模型：https://github.com/pytorch/vision/tree/master/torchvision/models
###### 知乎：PyTorch实战指南 https://zhuanlan.zhihu.com/p/29024978  

1、volatile和requires_grad: volatile=True相当于requires_grad=False,requires_grad=True 要求梯度  
2、y.data.norm():求y的标准差.三个2 .norm()结果，sqrt(2^2+2^2+2^2)   
3、pytorch的坑：https://www.jianshu.com/p/1fa86e060e5a  
4、transforms.Normalize：若tensor是三个维度，值在0到1之间变换到-1到1区间。transforms.Normalize((.5,.5,.5),(.5,.5,.5))  
5、iter() 函数：python函数，用来生成迭代器。  
6、pytorch加载模型GPU_CPU切换：torch.load('gen_500000.pkl', map_location=lambda storage, loc: storage.cuda(0))   
7、not callable错误：原因：在pytorch中定义网络的时候，采用逐层定义网络时，在结尾添加逗号，将定义的网络层转变成了元组。解决方案：去掉结尾逗号。  
8、Pytorch在训练过程中常见的问题：https://oldpan.me/archives/pytorch-conmon-problem-in-training  
9、pytorch单边加0的padding方式：torch.nn.functional.pad(x, (0, 1, 0, 1), 'constant', 0)  
10、加减层： http://www.cnblogs.com/marsggbo/p/8781774.html  
11、list转换成tensor: torch.cat(list) 或 torch.stack(list)  
12、数据指定位数输出：print('abc={0:.5f},{1:.2f}'.format(12.432145,23.456));结果abc=12.43215,23.46  
13、backward中retain_graph参数的作用：如果设置为False，计算图中的中间变量在计算完后就会被释放。  
14、torchvision.datasets.FashionMNIST中target_transform参数：数据集中对真值的变换。  
15、自定义forward：详情参考 https://pytorch.org/docs/stable/notes/extending.html  
16、tensor转置: 高维permute(),x = torch.randn(2, 3, 5),x.permute(2, 0, 1);两维transpose      
17、输出模型中间特征：参考https://github.com/lix3972/Python-Learning/blob/master/pytorch/code/FCN_easiest_master/FCN.py 中class VGGNet(VGG):……    
18、tensor常用语法：https://www.cnblogs.com/kk17/p/10252238.html   
19、tensor维度变化：https://www.cnblogs.com/taosiyu/p/11575005.html  
20、条件赋值：b[a>0] = 1  
21、给网络权重赋值：conv1.weight.data.fill_(1.0)，conv1.weight.data[0,0,1,1]=0或conv1.weight.data=a, 其中conv1=nn.Conv2d(1,1,3,1),a = torch.ones((1,1,3,3), dtype=torch.float),a[0,0,1,1]=0.0  
22、pytorch加载模型和初始化权重：https://www.jianshu.com/p/7a7d45b8e0ee
23、返回指定值在tensor中的index序号(编号/位置索引)：torch.nonzero((tmp[:, 0] == 2) * (tmp[:, 1] == 3))  
