###### 知乎：PyTorch实战指南 https://zhuanlan.zhihu.com/p/29024978

1、volatile和requires_grad: volatile=True相当于requires_grad=False,requires_grad=True 要求梯度  
2、y.data.norm():求y的标准差  
3、pytorch的坑：https://www.jianshu.com/p/1fa86e060e5a  
4、transforms.Normalize：设tensor是三个维度的，值在0到1之间变换到-1到1区间。transforms.Normalize((.5,.5,.5),(.5,.5,.5))
5、iter() 函数：python函数，用来生成迭代器。  
6、pytorch加载模型GPU_CPU切换：torch.load('gen_500000.pkl', map_location=lambda storage, loc: storage.cuda(0))   
7、not callable错误：不可调用。解决方案：去掉结尾逗号。

