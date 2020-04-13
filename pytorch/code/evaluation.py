import torch as th


# mask的batchsize必须是1, 或者就是3个维度 2个维度 统一变成3个维度。
def shape_cal(x):
    if len(x.shape) == 4:
        b, c, w, h = x.shape
        if b == 1:
            x = x.squeeze(0)
        else:
            print('batch size must be 1')
            return
    elif len(x.shape) == 3:
        c, w, h = x.shape
    elif len(x.shape) == 2:
        x = x.unsqueeze(0)
        c, w, h = x.shape
    else:
        print('The shape must be (1, c, w, h), (c, w, h), and (w, h).')
        return
    return x, c, w, h


def unnorm(x):  # 输入必须是三个维度(c, w, h)
    c, w, h = x.shape
    mean_list, std_list = [], []
    for i in range(c):
        mean_list.append(0.5)
        std_list.append(0.5)
    mean = th.tensor(mean_list)
    std = th.tensor(std_list)
    # mean = th.tensor([0.5])
    # std = th.tensor([0.5])

    return (std * x.permute(1, 2, 0) + mean).permute(2, 0, 1)


expr = """
y = x
c, w, h = y.shape
'mean = th.tensor([{}])'.format(('0.5, '*c)[:-2])
'std = th.tensor([{}])'.format(('0.5, ' * c)[:-2])
y = (std * x.permute(1, 2, 0) + mean).permute(2, 0, 1)
"""


# 遇到mask标准化的问题
def seg_eval(mask_pre, mask_gt):
    """
    :param mask_pre: type(mask_pre)=torch.tensor, shape=(1, c, w, h) or (c, w, h), 预测mask
    :param mask_gt: mask真值,type,shape同上
    :return: 语义分割评价标准PA：正确分类点数/像素点总数
    """
    mask_pre, c_pre, w_pre, h_pre = shape_cal(mask_pre)
    mask_gt, c_gt, w_gt, h_gt = shape_cal(mask_gt)
    mask_pre = unnorm(mask_pre)
    mask_gt = unnorm(mask_gt)
    # 统一都变成一个通道 值为0, 1, 2 …… 的情况，值即为分类编号
    # 注：真值mask_gt的值为整数 当其为1个通道时,就是非零即一的情况。
    # 但预测mask_pre是小数 当其为1个通道时需要取阀值使其非零即一,多通道时某位置的在多个通道上的最大值对应的层编号为该位置的值
    if c_gt > 1:  # 真值 分成多个通道
        mask_gt = th.argmax(mask_gt, 0)
    if c_pre > 1:  # 预测值
        mask_pre = th.argmax(mask_pre, 0)  # mask0: 1 chanel, value: 0, 1, 2, ...
    else:  # 预测mask是一个通道 但是值不是整数的情况
        mask_pre = mask_pre > mask_pre.max() / 3  # 单通道mask时 值大于最大值的1/3就认定为1. 可调整为1/2或其他
    mask_gt = mask_gt.to(dtype=th.int)
    mask_pre = mask_pre.to(dtype=th.int)
    # 将统一好的mask在变回多通道的情况 此时mask_pre也均为整数了
    obj_ids = th.unique(mask_gt)  # obj_ids对于 mask_pre也是一样
    # obj_ids = obj_ids.to(dtype=th.int)
    masks_gt = mask_gt == obj_ids[:, None, None]  # dtype = torch.uint8, shape = (class, w, h),class包括背景0
    masks_pre = mask_pre == obj_ids[:, None, None]  # dtype = torch.uint8

    # 计算PA
    # pii = mask_pre == mask_gt
    # PA = pii.sum().item() / (w_pre*h_pre)  # PA(Pixel Accuracy)

    # PA(Pixel Accuracy), MPA(Mean PA), IOU(Intersection over Union)
    pii_pij = 0
    intersection = 0
    union = 0
    # ####################################################
    # 既然obj_ids是mask_gt含有的值 怎么会有一项为0 该项为0与含有该项的值是矛盾的
    # ####################################################
    # sub_class_num = 0  # 应对mask中少一项 既然obj_ids是mask_gt含有的值 怎么会有一项为0 该项为0与含有该项的值是矛盾的
    for i in obj_ids:
        # pytorch没有找到逻辑与运算 用两步代替
        logic0 = masks_pre[i] + masks_gt[i]  # 都为1的地方相加=2, 1和0的位置=1, 0和0的位置=0
        p_and = logic0 > 1  # >1即=2的 为逻辑与的结果,也是正确分类的部分pii
        # 求或
        p_or = logic0 > 0  # >0 为逻辑或的结果
        # pij_sum = masks_gt[i].sum().item() 注：不加item()会导致计算结果异常
        # 需要处理 分母为0 的情况
        # 计算pii_pij分母为零 masks_gt[i].sum().item()=0意味着该图片不含该分类 求平均时应该算少一个分类
        # 计算Iou分母为零 union=0意味着预测值与真值同时为零 是预测准确的情况
        # if masks_gt[i].sum().item() == 0:
        #     pii_pij += 0
        #     sub_class_num += 1
        # else:
        pii_pij += p_and.sum().item() / masks_gt[i].sum().item()

        intersection += p_and.sum().item()
        # union += masks_pre[i] + masks_gt[i] - pii
        union += p_or.sum().item()
    PA = intersection / (w_pre*h_pre)
    MPA = pii_pij / obj_ids.shape[0]
    if union == 0:
        IoU = 0
    else:
        IoU = intersection / union

    return PA, MPA, IoU


if __name__ == '__main__':
    # s相当于mask_gt; (s0,s1,s2)是mask_gt分为3层的结果; p是mask_pre(按最大值)合为1层的效果
    # (p0,p1,p2)是mask_pre的三层;(y0,y1,y2)是mask_pre合为1层(取最大值)再分成3层的结果;
    s = [[0, 0, 1, 0], [1, 1, 1, 0], [1, 1, 2, 2], [0, 2, 2, 2]]
    p = [[0, 0, 1, 1], [1, 1, 1, 2], [1, 1, 2, 0], [2, 2, 2, 0]]

    p0 = [[0.9, 0.8, 0.3, 0.5], [0.1, 0.2, 0.5, 0.8], [0.2, 0.1, 0.1, 0.2], [0.3, 0.3, 0.5, 0.8]]
    s0 = [[1,     1,   0,   1],   [0,   0,   0,   1], [0,     0,   0,   0], [1,     0,    0,  0]]
    y0 = [[1,     1,   0,   0],   [0,   0,   0,   0], [0,     0,   0,   1], [0,     0,    0,  1]]

    p1 = [[0.0, 0.1, 0.5, 0.5], [0.8, 0.9, 0.7, 0.0], [0.3, 0.8, 0.0, 0.1], [0.0, 0.4, 0.2, 0.0]]
    s1 = [[0,    0,   1,   0],  [1,     1,   1,   0], [1,     1,   0,   0],  [0,    0,   0,   0]]
    y1 = [[0,    0,   1,   1],  [1,     1,   1,   0], [1,     1,   0,   0],  [0,    0,   0,   0]]

    p2 = [[0.0, 0.0, 0.3, 0.0], [0.1, 0.2, 0.3, 0.9], [0.0, 0.0, 0.9, 0.1], [0.4, 0.5, 0.6, 0.7]]
    s2 = [[0,     0,   0,   0], [0,     0,   0,   0], [0,     0,   1,   1], [0,     1,   1,   1]]
    y2 = [[0,     0,   0,   0], [0,     0,   0,   1], [0,     0,   1,   0], [1,     1,   1,   0]]

    gt = th.tensor(s)
    pre = th.tensor([p0, p1, p2])
    PA, MPA, IoU = seg_eval(pre, gt)
    print('PA={}, MPA={}, IoU={}'.format(PA, MPA, IoU))
    print('ok')
