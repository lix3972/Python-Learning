# 各程序文件功能
## affine_fucntion.py
pytorch 0.4版本的(1.0版本的都打成包看不到代码了) 仿射变换的代码，定义的反向传播函数(backward)有待研究，程序来自库文件vision.py。  
反向传播的return对应正向传播(forward)的输入，而反向传播的输入对应正向传播的输出。    
ctx个人理解：对于非tensor(不需要梯度的变量)相当于一个该类中的全局缓存变量，作用类似于self。对于tensor(需要梯度的变量)也是可以用其存储，但是需要用对应的函数进行调用。例如:在forward中ctx.save_for_backward(input, weight, bias)存储，在bakward中input, weight, bias = ctx.saved_tensors读取。还有ctx.needs_input_grad时一个布尔元组，指出输入(ctx.save_for_backward中的tensor)是否需要梯度计算(tensor对应的requires_grad属性是否为真)，。详情可以参考 https://pytorch.org/docs/stable/notes/extending.html  
## extract_mask.py
读取RGBA图片，分离R,G,B,alp四个通道，alp变为tensor，条件赋值变为二值矩阵(0和255),用plt.imshow(mask.numpy(), cmap='gray')显示为灰度图。  
## FCN_easiest_master
全连接网络进行语义分割，主要借鉴其中将VGG网络的中间变量输出的方法，还有onehot.py函数。  
## homo_transform.py

