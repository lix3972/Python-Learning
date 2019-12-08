主要参考：https://blog.csdn.net/zym19941119/article/details/80241663  
看似比较全面：https://www.cnblogs.com/q735613050/p/8969452.html(还没仔细看过)  
MS COCO 数据集主页(连不上)：http://mscoco.org/   

下面是可执行的python程序：

    import numpy as np
    import os
    from PIL import Image
    from matplotlib import pyplot as plt

    """
    原文链接： https://blog.csdn.net/zym19941119/article/details/80241663
    COCO数据集共有小类80个，分别为

        [‘person’, ‘bicycle’, ‘car’, ‘motorcycle’, ‘airplane’, ‘bus’, ‘train’, ‘truck’, ‘boat’, ‘traffic light’,
        ‘fire hydrant’, ‘stop sign’, ‘parking meter’, ‘bench’, ‘bird’, ‘cat’, ‘dog’, ‘horse’, ‘sheep’, ‘cow’,
        ‘elephant’, ‘bear’, ‘zebra’, ‘giraffe’, ‘backpack’, ‘umbrella’, ‘handbag’, ‘tie’, ‘suitcase’, ‘frisbee’,
        ‘skis’, ‘snowboard’, ‘sports ball’, ‘kite’, ‘baseball bat’, ‘baseball glove’, ‘skateboard’, ‘surfboard’,
        ‘tennis racket’, ‘bottle’, ‘wine glass’, ‘cup’, ‘fork’, ‘knife’, ‘spoon’, ‘bowl’, ‘banana’, ‘apple’,
        ‘sandwich’, ‘orange’, ‘broccoli’, ‘carrot’, ‘hot dog’, ‘pizza’, ‘donut’, ‘cake’, ‘chair’, ‘couch’,
        ‘potted plant’, ‘bed’, ‘dining table’, ‘toilet’, ‘tv’, ‘laptop’, ‘mouse’, ‘remote’, ‘keyboard’,
        ‘cell phone’, ‘microwave’, ‘oven’, ‘toaster’, ‘sink’, ‘refrigerator’,‘book’, ‘clock’, ‘vase’,
        ‘scissors’, ‘teddy bear’, ‘hair drier’, ‘toothbrush’]

    大类12个，分别为

        [‘appliance’, ‘food’, ‘indoor’, ‘accessory’, ‘electronic’, ‘furniture’, ‘vehicle’, ‘sports’,
        ‘animal’, ‘kitchen’, ‘person’, ‘outdoor’]

    安装COCO api

    COCO api来自于github, 从github上clone即可， https://github.com/pdollar/coco
    clone下来后在命令行中把路径切换到该路径，输入
    python setup.py install
    即可，如果遇到错误，参考这两篇博主写的博客即可
    https://blog.csdn.net/gxiaoyaya/article/details/78363391
    https://blog.csdn.net/qq_32768743/article/details/8020242

    """

    from pycocotools.coco import COCO

    path_train = '/media/lix/Disk4T/datasets/Coco2017/train/train2017'
    path_anno = '/media/lix/Disk4T/datasets/Coco2017/train/annotations_trainval2017'
    path_instances = '/media/lix/Disk4T/datasets/Coco2017/train/annotations_trainval2017/annotations/instances_train2017.json'
    # path_anno = '/media/lix/Disk4T/datasets/Coco2017/train/annotations_trainval2017/annotations'
    coco_instances = COCO(path_instances)  # 不同任务使用不同的annfile
    coco = coco_instances  # 原文中用的coco
    """
    getCatIds(catNms=[], supNms=[], catIds=[])
    通过输入类别的名字、大类的名字或是种类的id，来筛选得到图片所属类别的id
    比如，我们想知道dog类的id是多少
    """
    # catIds = coco_instances.getCatIds(catNms=['dog'])
    # 可以一次获取多个类别的id, catNms是小类别中的名字, supNms是打类别中的名称,例如 .getCatIds(supNms=['outdoor'])
    catIds = coco_instances.getCatIds(catNms=['dog', 'person', 'bicycle'])

    """
    getImgIds(imgIds=[], catIds=[])
    通过图片的id或是所属种类的id得到图片的id
    上一步得到了catIds包含了dog、person、bicycle三个类别的id，我们就可以查询到那些包含有这些类别的图片的id
    """
    imgIds = coco_instances.getImgIds(catIds=catIds)  # len(imgIds)  # 112

    """
    loadImgs(ids=[])
    得到图片的id信息后，就可以用loadImgs得到图片的信息了
    在这里我们随机选取之前list中的一张图片
    """
    img = coco_instances.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
    """
    最终得到的img并不是一张numpy格式或是PIL格式的图片，而是一个字典，包含了我们找到的这个id所代表的图片的信息 
    """
    path_img = os.path.join(path_train, img['file_name'])
    img_ = Image.open(path_img).convert('RGB')
    plt.imshow(img_)  # img.show() 也可以

    """
    getAnnIds(imgIds=[], catIds=[], areaRng=[], iscrowd=None)
    通过输入图片的id、类别的id、实例的面积、是否是人群来得到图片的注释id
    我们想要在之前的图片中画出对之前给定的三个种类进行实例分割的结果，就需要找到这张图片的注释信息的id
    """
    annIds = coco_instances.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    """
    loadAnns(ids=[])
    通过注释的id，得到注释的信息
    """
    anns = coco_instances.loadAnns(annIds)
    """
    showAnns(anns)
    使用标注的信息画出来分割的结果
    """
    coco.showAnns(anns)



print('ok')
