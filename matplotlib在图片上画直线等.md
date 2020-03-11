# matplotlib examples
## 1. 在RGB图上画线--程序测试可用  
参考：https://matplotlib.org/gallery/index.html   
读图片：  

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    import numpy as np

    im = np.array(Image.open('/home/lix/myCIDNN3/data/frame/000000.jpg'), dtype=np.uint8)
    # Create figure and axes
    fig, ax = plt.subplots(1)    
### 1.1 画直线
参考：https://matplotlib.org/gallery/userdemo/connect_simple01.html#sphx-glr-gallery-userdemo-connect-simple01-py

    line = patches.ConnectionPatch([100, 100], [200, 200], "data", "data")
    ax.add_patch(line)
### 1.2 画方框
    rect = patches.Rectangle((50,100),40,30,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
### 1.3 最后画图
    ax.imshow(im)
    plt.show()
    
## 2. 鼠标划线--待测试
参考：https://matplotlib.org/users/event_handling.html

    class LineBuilder:
        def __init__(self, line):
            self.line = line
            self.xs = list(line.get_xdata())
            self.ys = list(line.get_ydata())
            self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

        def __call__(self, event):
            print('click', event)
            if event.inaxes!=self.line.axes: return
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('click to add points')
    line, = ax.plot([], [], linestyle="none", marker="o", color="r")
    linebuilder = LineBuilder(line)

    plt.show()
    
程序源于：https://stackoverflow.com/questions/42578560/can-mouse-be-used-as-paintbrush-with-matplotlib

## 3. 形状与路径——patches与path
参考：https://blog.csdn.net/qq_27825451/article/details/82967904   
patches官方参考：https://matplotlib.org/api/patches_api.html#module-matplotlib.patches   
path官方参考：https://matplotlib.org/api/path_api.html?highlight=path#module-matplotlib.path    
个人理解：pathches是已经规定好的各种形状，path可以自由发挥的形状。path规定好形状后，结合pathches将形状添加到图形中。   

