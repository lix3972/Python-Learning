1、volatile和requires_grad: volatile=True相当于requires_grad=False,requires_grad=True 要求梯度  
2、y.data.norm():求y的标准差  
3、pytorch的坑：https://www.jianshu.com/p/1fa86e060e5a  
4、transforms.Normalize：设tensor是三个维度的，值在0到1之间变换到-1到1区间。transforms.Normalize((.5,.5,.5),(.5,.5,.5))