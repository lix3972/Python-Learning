# Python-Learning
## python学习网站  
1.https://docs.python.org/library/  Python库参考手册，可以选择python版本(目前默认3.7).https://docs.python.org/3/library/  
2.https://docs.python.org/  Python文档网站，网站地址会自动变成https://docs.python.org/3/  
https://www.w3cschool.cn/tensorflow_python/tensorflow_python-k14x2nc7.html  
## tensorflow : https://tensorflow.google.cn/api_docs/python/tf
## pytorch : https://pytorch.org/tutorials/
官方中文文档 https://pytorch.apachecn.org/  

## python 学习笔记  
###### 可以通过自带文档获得帮助，例如：   
###### help(copy.copy)  #给出函数或类等的帮助   
###### print(copy.copy.__doc__)  #给出较为详细的说明   
###### print(copy.__file__)  #给出copy文件的位置，可直接阅读源代码掌握其使用方法，如果文件列出.pyc文件，打开相应的.py文件即可。  
###### 注意：打开标准库文件时，存在修改的风险，不要保存可能的修改。有些源代码是解释器的组成部分或是用C语言编写的，可能无法读懂。
1、Python:是一种广泛意义上的编程语言。它非常适合做交互式的工作，并且足够强大可做大型应用。  
2、Numpy:是python的一个扩展，它定义了数组和矩阵，以及作用在它们上面的基本操作。  
3、Scipy:是另一个python的扩展包。它利用numpy做更高级的数学，信号处理，优化，统计等等。  
4、Matplotlib:是一个作图的python扩展包。  
5、argparse模块:是python标准库里面用来处理命令行参数的库。获取命令行参数。例如：python *.py --参数。  
6、termcolor是一个python包，可以改变控制台输出的颜色，支持各种terminal（WINDOWS的cmd.exe除外）。  
7、Scipy库构建于NumPy之上，提供了一个用于在Python中进行科学计算的工具集，如数值计算的算法和一些功能函数，可以方便的处理数据。  
8、tf.ConfigProto()配置Session运行参数&&GPU设备指定.  
9、a.strip().split(): strip()删除字符串a中开头和结尾指定字符，默认删除空格。split()按照给定的字符串分割字符串a。  
10、tf.stack()和tf.unstack()的用法,tf.stack（）这是一个矩阵拼接的函数，tf.unstack（）则是一个矩阵分解的函数.  
11、tf.transpose 多维的矩阵转置  
12、assert断言语句，声明其布尔值必须为真的判定，如果发生异常就说明表达示为假。  
13、tile() 平铺之意，用于在同一维度上的复制,进行张量扩展  
14、expand_dim() 增加维度  
15、object at 0x000002463192BAC8  法一：data=[i for i in zip(*I)]   法二：list(zip(*I))  
16、iter() 函数用来生成迭代器， .next()调用其中的值    
17、pycharm编译出现139错误：matplotlib与pandas版本的问题，更新matplotlib版本后解决。    
18、Pytorch用GPU加速：数据，网络，loss函数加.cuda() 例如：数据=torch.Tensor(数据).cuda();网路=网络.cuda();loss=loss.cuda()  
19、skimage导入：在pycharm中，安装scikit-image包。  
20、cuda error out of memory:(待解决)1)减少变量个数：用后面的变量替换前面的变量，变量用完后del掉(貌似用clear之类的，未尝试)，等。2）据说pytroch可以用checkpoint把参数分成两部分分别计算(会减慢速度，未尝试)。3）减少循环的使用。  
21、pytorch中transform函数:https://pytorch-cn.readthedocs.io/zh/latest/torchvision/torchvision-transform/  
22、python __getitem__()方法理解：当实例对象做P[key]运算时，就会调用类中的__getitem__()方法。   
23、pytorch中的torch.utils.data.Dataset和torch.utils.data.DataLoader: https://blog.csdn.net/geter_CS/article/details/83378786  
24、PIL.Image与numpy.array之间的相互转换：img = numpy.array(im);img = Image.fromarray(img.astype('uint8')).convert('RGB')    
25、sorted排序：mseAll.append((modelNumber,mseAB));  sortedMse=sorted(mseAll, key=lambda mse: mse[1])  ;  
按照'img_'和'.'中间的数值进行排序：self.A_paths = sorted(self.A_paths, key=lambda n: int(n.split('img_')[1].split('.')[0]))   
26、txt文件创建：f=open('txtName','a');f.write('Hello.\n');f.close();\n 表示换行。若：f.write('Hello\nworld'),则Hello与world分两行。'r' 只读模式,'w' 清空文件再重新写入,'a' 在文件末尾写入           
27、pytroch中的torch.nn:https://blog.csdn.net/dgyuanshaofeng/article/details/80345103  
28、mat读取方法：  
29、系统时间： time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())   
30、python文件名和文件路径操作：https://www.cnblogs.com/yanglang/p/7610838.html; os.path.join(path1,path2)  
31、得到文件夹中的文件名后，按照文件名中的数字排序（IMG_1.jpg;IMG_2.jpg;……）：  
A_paths = sorted(A_paths, key=lambda n: int(n.split('IMG_')[1].split('.')[0]))    
32、linux软硬链接：软链接: ln –s 源文件 目标文件; 硬链接: ln 源文件 目标文件，没有参数-s  
33、plt.imshow(tmp)出现错误：'numpy.ndarray' object has no attribute 'mask'的解决方法：plt.imshow(tmp.tolist())  
34、os路径相关：文件是否存在 if not os.path.exists(dir_save_pic): os.makedirs(dir_save_pic); 生成文件路径os.path.join(dir_save, 'pic')   
35、pycharm-project不显示文件夹：file->settings->project stucture->右侧+Add Content Root,选中项目文件夹，ok。  
36、各种paddle网络模型总结：https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/user_guides/models/index_cn.html#permalink-2-paddlecv  
37、CUDA_VISIBLE_DEVICES使用方法：系统环境参数，选择哪些gpu可见，哪些隐藏。可用os.environ['CUDA_VISIBLE_DEVICES']=gpu_id调整  
38、anaconda瘦身:conda clean -p 删除没用的包; conda clean -t 删除下载下来的 tar 包。   
39、once_differentiable作用：当你自定义的函数不可导时，在写backward函数时，就需要使用@once_differentiable。@staticmethod 及@once_differentiable用于不可导function的反向传播    
40、Vgg网络结构：https://blog.csdn.net/u014106644/article/details/89091949    
41、matplotlib：https://matplotlib.org/tutorials/  
42、os库_分离文件名和后缀：分离路径和文件:os.path.split()；分离文件名和后缀：os.path.splitext()——详见os库常用函数总结.md。获取路径中的文件名os.path.basename(path);不要文件后缀名filename_nosuffix = filename.split('.')[0]  
43、膨胀与腐蚀算法：参考https://blog.csdn.net/whchang8/article/details/56834608 ； https://blog.csdn.net/qq_29540745/article/details/51931770    
44、自定义卷积网络前向后向传播：c 的 https://blog.csdn.net/g11d111/article/details/83021651 ； python的 https://blog.csdn.net/oBrightLamp/article/details/84589545 (代码存入pytorch/code/convolution.py)  
45、mat文件读取：import h5py；train_gt = h5py.File(fileName,'r')；train_gt = train_gt['outputD_map']；也可用import scipy.io as scio；data = scio.loadmat(data_path)；data_train_label=data_train.get('label')  
46、wget安装：遇到 No module named 'wget'错误，安装命令：pip install wget  
47、PIL中灰度图convert问题：convert('L') 大写字母L.详见： https://github.com/lix3972/Python-Learning/blob/master/PIL_image.md   
48、多个网络学习率要保持一致，否则会loss = nan：在COVID19_3/train4_classify.py中使用。   
49、nunpy array数据读写状态：mask.flags.writeable = True 在COVID19_3/train4_classify.py中class Covid19Dataset4(object)使用。  
50、网络互不影响时detach()位置: output=net(input)后，下一步操作就在output.detach()。在COVID19_3/train4_classify.py中使用。   
51、nii文件读取：import nibabel as nib;imgs = nib.load(path_imgs);imgs = imgs.dataobj[:, :, num]。在COVID19中定义dataset使用。   
