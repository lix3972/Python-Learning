有待研究的代码
#### affine_fucntion.py
pytorch 0.4版本的(1.0版本的都打成包看不到代码了) 仿射变换的代码，定义的反向传播函数(backward)有待研究，程序来自库文件vision.py。  
反向传播的return对应正向传播(forward)的输入，而反向传播的输入对应正向传播的输出。    
ctx个人理解：对于非tensor(不需要梯度的变量)相当于一个该类中的全局缓存变量，作用类似于self。对于tensor(需要梯度的变量)也是可以用其存储，但是需要用对应的函数进行调用。例如:在forward中ctx.save_for_backward(input, weight, bias)存储，在bakward中input, weight, bias = ctx.saved_tensors读取。还有ctx.needs_input_grad[0](好像是输入的第一项)。详情可以参考 https://pytorch.org/docs/stable/notes/extending.html  