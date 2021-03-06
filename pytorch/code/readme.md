# 各程序文件功能
 
## FCN_easiest_master
全连接网络进行语义分割，主要借鉴其中将VGG网络的中间变量输出的方法，还有onehot.py函数。  
## PoissonImageEdit
计划将Poisson合成算法应用于网络中，需要能反向传播。
* get_contour.py 得到mask(非0即1)的边界，边界在mask上。
* get_outer_contour.py 得到的mask大一圈的边界，边界不在mask上，在mask外边一圈。
## homo_transform
单应性变换(投影变换)的程序，其中的dislodge_zero.py中的类DislodgeZero，是第一次定义带有反向传播函数的类，其目的是为了避免除以零(在homo_transform.py中需要除法)，其功能是当输入的绝对值小于forward中指定的数值时让其等于指定值，反向传播时当输入的绝对值小于forward中指定的数值时让反向传播数值等于指定值。
## affine_fucntion.py
pytorch 0.4版本的(1.0版本的都打成包看不到代码了) 仿射变换的代码，定义的反向传播函数(backward)有待研究，程序来自库文件vision.py。  
反向传播的return对应正向传播(forward)的输入，而反向传播的输入对应正向传播的输出。    
ctx个人理解：对于非tensor(不需要梯度的变量)相当于一个该类中的全局缓存变量，作用类似于self。对于tensor(需要梯度的变量)也是可以用其存储，但是需要用对应的函数进行调用。例如:在forward中ctx.save_for_backward(input, weight, bias)存储，在bakward中input, weight, bias = ctx.saved_tensors读取。还有ctx.needs_input_grad时一个布尔元组，指出输入(ctx.save_for_backward中的tensor)是否需要梯度计算(tensor对应的requires_grad属性是否为真)，。详情可以参考 https://pytorch.org/docs/stable/notes/extending.html  
## convolution.py
网络节选的程序，自定义卷积网络的前向和后向传播。  
## evaluation.py
自定义评价指标。seg_eval函数输出分割评价指标PA(Pixel Accuracy), MPA(Mean PA), IOU(Intersection over Union)
## extract_mask.py
读取RGBA图片，分离R,G,B,alp四个通道，alp变为tensor，条件赋值变为二值矩阵(0和255),用plt.imshow(mask.numpy(), cmap='gray')显示为灰度图。
## my_module.py
自定义前向、反向传播并带weights和bias的module，参数可更新。  
参考：https://pytorch.org/tutorials/beginner/nn_tutorial.html#      
## try_backward_grad.py
测试自定义类的反向传播  
## try_martix_grad.py
矩阵解方程Ax=b，已知A和b求x。为了测试是否可以反向传播，求x对A和b的梯度。
## two_layer_net_custom_function.py
* 可以看到ReLU()的定义过程，包括前向和后向传播的定义。   

Pytorch网站: https://github.com/pytorch/tutorials/blob/master/beginner_source/examples_autograd/two_layer_net_custom_function.py    
github：https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html#sphx-glr-beginner-examples-autograd-two-layer-net-custom-function-py    
Pytorch帮助文档：https://pytorch.org/tutorials/beginner/pytorch_with_examples.html 示例之一  
## two_layer_net_numpy.py  
源于pytorch官网上的一个例子，用numpy定义一个两层网络。程序添加了注释，有助于理解CNN网络的基本结构。
