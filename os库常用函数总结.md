## 从完整路径中获取目录、文件名、后缀名
    import os
    src_path = './try/source.png'
    path, file = os.path.split(src_path)  # 分离路径和文件path='./try', file='source.png'
    file_name, suffix = os.path.splitext(file)  # 分离文件名和后缀file_name=source, suffix=.png
获取路径中的文件名也可用 os.path.basename(path);
去掉文件后缀名也可用 filename_nosuffix = filename.split('.')[0]
## 文件是否存在/生成文件夹 
    if not os.path.exists(dir_save_pic): 
        os.makedirs(dir_save_pic); 
## 生成(合并)文件路径
os.path.join(dir_save, 'pic')
## 遍历文件夹获得文件夹下文件名
参考 https://www.runoob.com/python/os-walk.html
os.walk() 方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下。

walk()方法语法格式如下：

os.walk(top[, topdown=True[, onerror=None[, followlinks=False]]])

参数

    top -- 是你所要遍历的目录的地址, 返回的是一个三元组(root,dirs,files)。
        root 所指的是当前正在遍历的这个文件夹的本身的地址
        dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
        files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)

    topdown --可选，为 True，则优先遍历 top 目录，否则优先遍历 top 的子目录(默认为开启)。如果 topdown 参数为 True，walk 会遍历top文件夹，与top 文件夹中每一个子目录。

    onerror -- 可选，需要一个 callable 对象，当 walk 需要异常时，会调用。

    followlinks -- 可选，如果为 True，则会遍历目录下的快捷方式(linux 下是软连接 symbolic link )实际所指的目录(默认关闭)，如果为 False，则优先遍历 top 的子目录。

返回值

该方法没有返回值。可以paths = os.walk(alp_path)，paths是一个生成器,paths.__next__()返回三个list(root, dirs, files)  
实例  

以下实例演示了 walk() 方法的使用：  

        import os
        for root, dirs, files in os.walk(".", topdown=False):
            for name in files:
                print(os.path.join(root, name))
            for name in dirs:
                print(os.path.join(root, name))
