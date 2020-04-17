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
    """
        :param mask_pre: type(mask_pre)=torch.tensor, shape=(1, c, w, h) or (c, w, h), 预测mask
        :param mask_gt: mask真值,type,shape同上
        :return: 语义分割评价标准PA：正确分类点数/像素点总数; MPA, IoU; 只评价前景加后缀f,背景加b,全部a
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
    # obj_ids = obj_ids.to(dtype=th.int)

    masks_gt = mask_gt == obj_ids[:, None, None]  # dtype = torch.uint8, shape = (class, w, h),class包括背景0
    masks_pre = mask_pre == obj_ids[:, None, None]  # dtype = torch.uint8
    # 计算PA
    # pii = mask_pre == mask_gt
    # PA = pii.sum().item() / (w_pre*h_pre)  # PA(Pixel Accuracy)

    # PA(Pixel Accuracy), MPA(Mean PA), IOU(Intersection over Union)
    pii_pij_list = []
    intersection_list = []
    union_list = []

    # 在使用类似mask[i]方法时会出现其shape=(0, 1, 512, 512)并且mask[i].sum()=0的奇怪情况
    for i in range(obj_ids.shape[0]):
        ### print('masks_gt.shape={}, masks_pre.shape={}, obj_ids={}, i={}'.format(masks_gt.shape, masks_pre.shape, obj_ids, i))

        # pytorch没有找到逻辑与运算 用两步代替
        logic0 = masks_pre[i] + masks_gt[i]  # 都为1的地方相加=2, 1和0的位置=1, 0和0的位置=0
        p_and = logic0 > 1  # >1即=2的 为逻辑与的结果,也是正确分类的部分pii
        # 求或
        p_or = logic0 > 0  # >0 为逻辑或的结果
        # pij_sum = masks_gt[i].sum().item() 注：不加item()会导致计算结果异常
        pii_pij_list.append(p_and.sum().item() / masks_gt[i].sum().item())
        intersection_list.append(p_and.sum().item())
        union_list.append(p_or.sum().item())

    # 包含背景和前景
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


class EvalSegmentation(object):
    def __init__(self):
        self.PAfs, self.MPAfs, self.IoUfs = 0, 0, 0
        self.PAbs, self.MPAbs, self.IoUbs = 0, 0, 0
        self.PAs, self.MPAs, self.IoUs = 0, 0, 0
        self.num = 0
        self.sorted_PA, self.sorted_MPA, self.sorted_IoU, self.sorted_avg_eval3 = [], [], [], []
        self.sorted_PAf, self.sorted_MPAf, self.sorted_IoUf, self.sorted_avg_evalf3 = [], [], [], []
        self.sorted_PAb, self.sorted_MPAb, self.sorted_IoUb, self.sorted_avg_evalb3 = [], [], [], []

    def reset(self):
        self.PAfs, self.MPAfs, self.IoUfs = 0, 0, 0
        self.PAbs, self.MPAbs, self.IoUbs = 0, 0, 0
        self.PAs, self.MPAs, self.IoUs = 0, 0, 0
        self.num = 0

    @staticmethod
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

    @staticmethod
    def unnorm(x):  # 输入必须是三个维度(c, w, h)
        c, w, h = x.shape
        mean_list, std_list = [], []
        for i in range(c):
            mean_list.append(0.5)
            std_list.append(0.5)
        mean = th.tensor(mean_list)
        std = th.tensor(std_list)
        return (std * x.permute(1, 2, 0) + mean).permute(2, 0, 1)

    def seg_eval(self, mask_pre, mask_gt, gt_max=1, threshold_denominator=3, device='cpu', isretrun=False):
        # 一张图的评价指标 多个图的指标累加求平均 放在 for i, data in enumerate(data_loader): 里面
        if device == 'cpu':
            mask_pre = mask_pre.cpu()
            mask_gt = mask_gt.cpu()
        mask_pre, c_pre, w_pre, h_pre = self.shape_cal(mask_pre)
        mask_gt, c_gt, w_gt, h_gt = self.shape_cal(mask_gt)
        mask_pre = self.unnorm(mask_pre)
        mask_gt = self.unnorm(mask_gt)
        if c_gt > 1:  # 真值 分成多个通道
            mask_gt = th.argmax(mask_gt, 0)
        else:
            mask_gt = mask_gt * gt_max
        if c_pre > 1:  # 预测值
            mask_pre = th.argmax(mask_pre, 0)  # mask0: 1 chanel, value: 0, 1, 2, ...
        else:  # 预测mask是一个通道 但是值不是整数的情况
            mask_pre = mask_pre > mask_pre.max() / threshold_denominator  # 单通道mask时 值大于最大值的1/3就认定为1. 可调整为1/2或其他
        mask_gt = mask_gt.to(dtype=th.int)
        mask_pre = mask_pre.to(dtype=th.int)
        # 将统一好的mask在变回多通道的情况 此时mask_pre也均为整数了
        obj_ids = th.unique(mask_gt)  # obj_ids对于 mask_pre也是一样

        masks_gt = mask_gt == obj_ids[:, None, None]  # shape = (class, w, h),class包括背景0
        masks_pre = mask_pre == obj_ids[:, None, None]

        pii_pij_list = []
        intersection_list = []
        union_list = []

        for i in range(obj_ids.shape[0]):
            # 与运算
            logic0 = masks_pre[i] + masks_gt[i]  # 都为1的地方相加=2, 1和0的位置=1, 0和0的位置=0
            p_and = logic0 > 1  # >1即=2的 为逻辑与的结果,也是正确分类的部分pii
            # 求或
            p_or = logic0 > 0  # >0 为逻辑或的结果

            pii_pij_list.append(p_and.sum().item() / masks_gt[i].sum().item())
            intersection_list.append(p_and.sum().item())
            union_list.append(p_or.sum().item())
        # 包含背景和前景
        PA = sum(intersection_list) / (w_pre * h_pre)
        MPA = sum(pii_pij_list) / len(pii_pij_list)
        IoU = sum(intersection_list) / sum(union_list)
        # 不含背景 只有前景 f表示前景, 如果只有背景那么输出-1
        if len(obj_ids) > 1:
            PAf = sum(intersection_list[1:]) / masks_gt[1:].sum().item()
            MPAf = sum(pii_pij_list[1:]) / len(pii_pij_list[1:])
            IoUf = sum(intersection_list[1:]) / sum(union_list[1:])
        else:
            PAf = 0
            MPAf = 0
            IoUf = 0
        # 只有背景 b表示背景
        PAb = intersection_list[0] / masks_gt[0].sum().item()
        MPAb = pii_pij_list[0] / 1
        IoUb = intersection_list[0] / union_list[0]

        # 只统计前景f
        if len(obj_ids) > 1:  # 有些只有背景没有前景 obj_ids长度大于1表示有前景
            self.num += 1
            self.PAfs += PAf
            self.MPAfs += MPAf
            self.IoUfs += IoUf

        # 只统计背景b
        self.PAbs += PAb
        self.MPAbs += MPAb
        self.IoUbs += IoUb

        # 前景背景都包含
        self.PAs += PA
        self.MPAs += MPA
        self.IoUs += IoU

        if isretrun:
            return PAf, MPAf, IoUf, PAb, MPAb, IoUb, PA, MPA, IoU

    def avg_eval(self, modelNum, isretrun=False):
        # 统计多个图评价指标的平均
        # 累加的工作在seg_eval中完成  放在 for i, data in enumerate(data_loader): 外边
        # 其实每次都会统计累计指标的平均值

        # 计算平均值 然后 将平均值加入到列表
        # 只统计前景f
        if self.num > 0:
            self.PAfs_avg  = self.PAfs  / self.num
            self.MPAfs_avg = self.MPAfs / self.num
            self.IoUfs_avg = self.IoUfs / self.num
            self.avg_evalf3 = (self.PAfs_avg + self.MPAfs_avg + self.IoUfs_avg) / 3
        else:  # self.num <= 0 表示测试集中没有前景
            self.PAfs_avg = -1
            self.MPAfs_avg = -1
            self.IoUfs_avg = -1
            self.avg_evalf3 = -1
        self.sorted_PAf.append([self.PAfs_avg, modelNum])
        self.sorted_MPAf.append([self.MPAfs_avg, modelNum])
        self.sorted_IoUf.append([self.IoUfs_avg, modelNum])
        self.sorted_avg_evalf3.append([self.avg_evalf3, modelNum])

        # 只统计背景b
        self.PAbs_avg  = self.PAbs / self.num
        self.MPAbs_avg = self.MPAbs / self.num
        self.IoUbs_avg = self.IoUbs / self.num
        self.avg_evalb3 = (self.PAbs_avg + self.MPAbs_avg + self.IoUbs_avg) / 3
        self.sorted_PAb.append([self.PAbs_avg, modelNum])
        self.sorted_MPAb.append([self.MPAbs_avg, modelNum])
        self.sorted_IoUb.append([self.IoUbs_avg, modelNum])
        self.sorted_avg_evalb3.append([self.avg_evalb3, modelNum])

        # 前景背景都包含
        self.PAs_avg  = self.PAs  / self.num
        self.MPAs_avg = self.MPAs / self.num
        self.IoUs_avg = self.IoUs / self.num
        self.avg_eval3 = (self.PAs_avg + self.MPAs_avg + self.IoUs_avg) / 3
        self.sorted_PA.append([self.PAs_avg, modelNum])
        self.sorted_MPA.append([self.MPAs_avg, modelNum])
        self.sorted_IoU.append([self.IoUs_avg, modelNum])
        self.sorted_avg_eval3.append([self.avg_eval3, modelNum])

        if isretrun:
            return self.PAfs_avg, self.MPAfs_avg, self.IoUfs_avg, self.PAbs_avg, self.MPAbs_avg, self.IoUbs_avg, self.PAs_avg, self.MPAs_avg, self.IoUs_avg

    def show_avg_eval(self, modelNum, isretrun=True):
        # 显示评价指标 放在avg_eval 后边
        message_f = 'model {}: Forground only: PAf={}, MPAf={}, Iouf={}'.format(modelNum, self.PAfs_avg, self.MPAfs_avg, self.IoUfs_avg)
        message_b = 'model {}: Background only: PAb={}, MPAb={}, Ioub={}'.format(modelNum, self.PAbs_avg, self.MPAbs_avg, self.IoUbs_avg)
        message_a = 'model {}: All: PA={}, MPA={}, Iou={}'.format(modelNum, self.PAs_avg, self.MPAs_avg, self.IoUs_avg)
        print(message_f)
        print(message_b)
        print(message_a)
        if isretrun:
            return message_f, message_b, message_a

    def show_sorted_models_eval(self, title='', top=3, isretrun=True):
        # 显示多个模型在整个数据集中的平均值的排序 放在for modelNum in modelNums: 外边
        message_f = '{} Forground only: sorted_PA={}, sorted_MPA={}, sorted_Iou={}, sorted_avg_eval3={}'.format(
            title, sorted(self.sorted_PAf)[-top:], sorted(self.sorted_MPAf)[-top:], sorted(self.sorted_IoUf)[-top:],
            sorted(self.sorted_avg_evalf3)[-top:])
        print(message_f)

        message_b = '{}: Background only: sorted_PA={}, sorted_MPA={}, sorted_Iou={}, sorted_avg_eval3={}'.format(
            title, sorted(self.sorted_PAb)[-top:], sorted(self.sorted_MPAb)[-top:], sorted(self.sorted_IoUb)[-top:],
            sorted(self.sorted_avg_evalb3)[-top:])
        print(message_b)

        message_a = '{}: all: sorted_PA={}, sorted_MPA={}, sorted_Iou={}, sorted_avg_eval3={}'.format(
            title, sorted(self.sorted_PA)[-top:], sorted(self.sorted_MPA)[-top:], sorted(self.sorted_IoU)[-top:],
            sorted(self.sorted_avg_eval3)[-top:])
        print(message_a)

        if isretrun:
            return message_f, message_b, message_a


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
    PAf, MPAf, IoUf, PAb, MPAb, IoUb, PA, MPA, IoU = seg_eval(pre_norm, gt_norm, gt.max().item())
    print('all: PA={}, MPA={}, IoU={}'.format(PA, MPA, IoU))
    print('fore: PAf={}, MPAf={}, IoUf={}'.format(PAf, MPAf, IoUf))
    print('back: PAb={}, MPAb={}, IoUb={}'.format(PAb, MPAb, IoUb))

    print('ok')
