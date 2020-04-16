import torch as th
import torchvision.transforms as transforms


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

    return (std * x.permute(1, 2, 0) + mean).permute(2, 0, 1)


# 遇到mask标准化的问题 如果mask就是一个背景一个前景,gt_max=1.如果多个前景gt_max=不包括背景的分类数
def seg_eval(mask_pre, mask_gt, gt_max=1):
    mask_pre, c_pre, w_pre, h_pre = shape_cal(mask_pre)
    mask_gt, c_gt, w_gt, h_gt = shape_cal(mask_gt)
    mask_pre = unnorm(mask_pre)
    mask_gt = unnorm(mask_gt)
    # 统一都变成一个通道 值为0, 1, 2 …… 的情况，值即为分类编号
    # 注：真值mask_gt的值为整数 当其为1个通道时,就是非零即一的情况。
    # 但预测mask_pre是小数 当其为1个通道时需要取阀值使其非零即一,多通道时某位置的在多个通道上的最大值对应的层编号为该位置的值
    if c_gt > 1:  # 真值 分成多个通道
        mask_gt = th.argmax(mask_gt, 0)
    else:
        mask_gt = mask_gt * gt_max  # mask_gt多个这都归一化到0～1,乘以gt_max在扩到多个值.
    if c_pre > 1:  # 预测值
        mask_pre = th.argmax(mask_pre, 0)  # mask0: 1 chanel, value: 0, 1, 2, ...
    else:  # 预测mask是一个通道 但是值不是整数的情况
        mask_pre = mask_pre > mask_pre.max() / 3  # 单通道mask时 值大于最大值的1/3就认定为1. 可调整为1/2或其他
    mask_gt = mask_gt.to(dtype=th.int)
    mask_pre = mask_pre.to(dtype=th.int)
    # 将统一好的mask在变回多通道的情况 此时mask_pre也均为整数了
    obj_ids = th.unique(mask_gt)  # obj_ids对于 mask_pre也是一样

    masks_gt = mask_gt == obj_ids[:, None, None]  # dtype = torch.uint8, shape = (class, w, h),class包括背景0
    masks_pre = mask_pre == obj_ids[:, None, None]  # dtype = torch.uint8

    pii_pij_list = []
    intersection_list = []
    union_list = []

    for i in range(obj_ids.shape[0]):
        ### print('masks_gt.shape={}, masks_pre.shape={}, obj_ids={}, i={}'.format(masks_gt.shape, masks_pre.shape, obj_ids, i))

        # pytorch没有找到逻辑与运算 用两步代替
        logic0 = masks_pre[i] + masks_gt[i]  # 都为1的地方相加=2, 1和0的位置=1, 0和0的位置=0
        p_and = logic0 > 1  # >1即=2的 为逻辑与的结果,也是正确分类的部分pii
        # 求或
        p_or = logic0 > 0  # >0 为逻辑或的结果

        pii_pij_list.append(p_and.sum().item() / masks_gt[i].sum().item())
        intersection_list.append(p_and.sum().item())
        union_list.append(p_or.sum().item())

    PA = sum(intersection_list) / (w_pre*h_pre)
    MPA = sum(pii_pij_list) / len(pii_pij_list)
    IoU = sum(intersection_list) / sum(union_list)
    # 不含背景 只有前景 f表示前景, 如果只有背景那么输出-1
    if len(obj_ids) > 1:
        PAf = sum(intersection_list[1:]) / masks_gt[1:].sum().item()
        MPAf = sum(pii_pij_list[1:]) / len(pii_pij_list[1:])
        IoUf = sum(intersection_list[1:]) / sum(union_list[1:])
    else:
        PAf = -1
        MPAf = -1
        IoUf = -1
    # 只有背景 b表示背景
    PAb = intersection_list[0] / masks_gt[0].sum().item()
    MPAb = pii_pij_list[0] / 1
    IoUb = intersection_list[0] / union_list[0]

    return PAf, MPAf, IoUf, PAb, MPAb, IoUb, PA, MPA, IoU


if __name__ == '__main__':
    # s相当于mask_gt; (s0,s1,s2)是mask_gt分为3层的结果; p是mask_pre(按最大值)合为1层的效果
    # (p0,p1,p2)是mask_pre的三层;(y0,y1,y2)是mask_pre合为1层(取最大值)再分成3层的结果;
    # 类别  分类正确个数  gt个数  pre并gt个数
    # 0     2           5       7
    # 1     6           6       7
    # 2     3           5       7
    # PA = (2+6+3)/16=11/16=0.6875 ;         PAf = (6+3)/(6+5)=9/11=0.81818 ; PAb = 2/5=0.4
    # MPA = (2/5 + 6/6 + 3/5)/3=2/3=0.6667 ; MPAf = (6/6 + 3/5)/2=1.6/2=0.8 ; MPAb = 2/5=0.4
    # IoU = (2+6+3)/(7+7+7)=11/21=0.5238 ;   IoUf = (6+3)/(7+7)=9/14=0.6428 ; IoUb = 2/7=0.2857
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

    gt = th.tensor(s).unsqueeze(0).to(dtype=th.float)
    pre = th.tensor([p0, p1, p2]).to(dtype=th.float)
    gt_norm = transforms.Normalize((0.5,), (0.5,))(gt/gt.max())
    pre_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(pre)
    PAf, MPAf, IoUf, PAb, MPAb, IoUb, PA, MPA, IoU = seg_eval(pre_norm, gt_norm, gt.max())
    print('all: PA={}, MPA={}, IoU={}'.format(PA, MPA, IoU))
    print('fore: PAf={}, MPAf={}, IoUf={}'.format(PAf, MPAf, IoUf))
    print('back: PAb={}, MPAb={}, IoUb={}'.format(PAb, MPAb, IoUb))
    print('ok')
