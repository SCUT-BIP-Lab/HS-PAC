# hand_eval_utils.py
"""
Hand mesh/pose evaluation utilities, migrated from mobrecon/runner.py and its dependencies.
"""
import numpy as np
import torch
import cv2
from scipy.optimize import minimize

# ---- EvalUtil ----
class EvalUtil:
    """ Util class for evaluation networks. """
    def __init__(self, num_kp=21):
        self.data = list()
        self.num_kp = num_kp
        for _ in range(num_kp):
            self.data.append(list())
    def feed(self, keypoint_gt, keypoint_pred, keypoint_vis=None):
        if isinstance(keypoint_gt, torch.Tensor):
            keypoint_gt = keypoint_gt.detach().cpu().numpy()
        if isinstance(keypoint_pred, torch.Tensor):
            keypoint_pred = keypoint_pred.detach().cpu().numpy()
        keypoint_gt = np.squeeze(keypoint_gt)
        keypoint_pred = np.squeeze(keypoint_pred)
        if keypoint_vis is None:
            keypoint_vis = np.ones_like(keypoint_gt[:, 0])
        keypoint_vis = np.squeeze(keypoint_vis).astype("bool")
        assert len(keypoint_gt.shape) == 2
        assert len(keypoint_pred.shape) == 2
        assert len(keypoint_vis.shape) == 1
        diff = keypoint_gt - keypoint_pred
        euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=1))
        num_kp = keypoint_gt.shape[0]
        for i in range(num_kp):
            if keypoint_vis[i]:
                self.data[i].append(euclidean_dist[i])
    def _get_pck(self, kp_id, threshold):
        if len(self.data[kp_id]) == 0:
            return None
        data = np.array(self.data[kp_id])
        pck = np.mean((data <= threshold).astype("float"))
        return pck
    def get_pck_all(self, threshold):
        pckall = []
        for kp_id in range(self.num_kp):
            pck = self._get_pck(kp_id, threshold)
            pckall.append(pck)
        pckall = np.mean(np.array(pckall))
        return pckall
    def _get_epe(self, kp_id):
        if len(self.data[kp_id]) == 0:
            return None, None
        data = np.array(self.data[kp_id])
        epe_mean = np.mean(data)
        epe_median = np.median(data)
        return epe_mean, epe_median
    def get_measures(self, val_min, val_max, steps):
        thresholds = np.linspace(val_min, val_max, steps)
        thresholds = np.array(thresholds)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)
        epe_mean_all = list()
        epe_median_all = list()
        auc_all = list()
        pck_curve_all = list()
        for part_id in range(self.num_kp):
            mean, median = self._get_epe(part_id)
            if mean is None:
                continue
            epe_mean_all.append(mean)
            epe_median_all.append(median)
            pck_curve = list()
            for t in thresholds:
                pck = self._get_pck(part_id, t)
                pck_curve.append(pck)
            pck_curve = np.array(pck_curve)
            pck_curve_all.append(pck_curve)
            auc = np.trapz(pck_curve, thresholds)
            auc /= norm_factor
            auc_all.append(auc)
        epe_mean_joint = epe_mean_all
        epe_mean_all = np.mean(np.array(epe_mean_all))
        epe_median_all = np.mean(np.array(epe_median_all))
        auc_all = np.mean(np.array(auc_all))
        pck_curve_all = np.mean(np.array(pck_curve_all), axis=0)
        return (
            epe_mean_all,
            epe_mean_joint,
            epe_median_all,
            auc_all,
            pck_curve_all,
            thresholds,
        )

# ---- Rigid Alignment ----
def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))
    varP = np.var(A, axis=0).sum()
    c = 1/varP * np.sum(s)
    t = -np.dot(c*R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t

def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c*R, np.transpose(A))) + t
    return A2

# ---- Kinematics: MANO/MPII mapping ----
class MANOHandJoints:
    n_keypoints = 21
    n_joints = 21
    center = 4
    root = 0
    labels = [
        'W', 'I0', 'I1', 'I2', 'M0', 'M1', 'M2', 'L0', 'L1', 'L2',
        'R0', 'R1', 'R2', 'T0', 'T1', 'T2', 'I3', 'M3', 'L3', 'R3', 'T3'
    ]
    mesh_mapping = {16: 333, 17: 444, 18: 672, 19: 555, 20: 744}
    parents = [None, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 3, 6, 9, 12, 15]
    end_points = [0, 16, 17, 18, 19, 20]
class MPIIHandJoints:
    n_keypoints = 21
    n_joints = 21
    center = 9
    root = 0
    labels = [
        'W', 'T0', 'T1', 'T2', 'T3', 'I0', 'I1', 'I2', 'I3',
        'M0', 'M1', 'M2', 'M3', 'R0', 'R1', 'R2', 'R3', 'L0', 'L1', 'L2', 'L3'
    ]
    parents = [None, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
    colors = [
        (0, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0),
        (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),
        (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255),
        (255, 255, 0), (255, 255, 0), (255, 255, 0), (255, 255, 0),
        (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255),
    ]
def mano_to_mpii(mano):
    mpii = []
    for j in range(MPIIHandJoints.n_joints):
        mpii.append(mano[MANOHandJoints.labels.index(MPIIHandJoints.labels[j])])
    mpii = np.stack(mpii, 0)
    return mpii

def mpii_to_mano(mpii):
    mano = []
    for j in range(MANOHandJoints.n_joints):
        mano.append(mpii[MPIIHandJoints.labels.index(MANOHandJoints.labels[j])])
    mano = np.stack(mano, 0)
    return mano

# ---- Visual/Mask metrics ----
def compute_iou(pred, gt):
    area_pred = pred.sum()
    area_gt = gt.sum()
    if area_pred == area_gt == 0:
        return 1
    union_area = (pred + gt).clip(max=1)
    union_area = union_area.sum()
    inter_area = area_pred + area_gt - union_area
    IoU = inter_area / union_area
    return IoU

def cnt_area(cnt):
    area = cv2.contourArea(cnt)
    return area

# ---- Registration (2D-1D) ----
def perspective_np(points, calibrations):
    if points.shape[1] == 2:
        points = np.concatenate([points, np.ones([points.shape[0], 1])], -1)
    z = points[:, 2:3].copy()
    points[:, :3] /= z
    points1 = np.concatenate([points, np.ones([points.shape[0], 1])], -1)
    points_img = np.dot(calibrations, points1.T).T
    points_img = np.concatenate([points_img[:, :2], z], -1)
    return points_img

def registration(vertex, uv, j_regressor, calib, size, uv_conf=None, poly=None):
    t = np.array([0, 0, 0.6])
    bounds = ((None, None), (None, None), (0.05, 2))
    poly_protect = [0.06, 0.02]
    vertex2xyz = mano_to_mpii(np.matmul(j_regressor, vertex))
    try_poly = True
    if uv_conf is None:
        uv_conf = np.ones([uv.shape[0], 1])
    uv_select = uv_conf > 0.1
    if uv_select.sum() == 0:
        success = False
    else:
        loss = np.array([5, ])
        attempt = 5
        while loss.mean() > 2 and attempt:
            attempt -= 1
            uv = uv[uv_select.repeat(2, axis=1)].reshape(-1, 2)
            uv_conf = uv_conf[uv_select].reshape(-1, 1)
            vertex2xyz = vertex2xyz[uv_select.repeat(3, axis=1)].reshape(-1, 3)
            sol = minimize(align_uv, t, method='SLSQP', bounds=bounds, args=(uv, vertex2xyz, calib))
            t = sol.x
            success = sol.success
            xyz = vertex2xyz + t
            proj = perspective_np(xyz, calib)[:, :2]
            loss = abs((proj - uv).sum(axis=1))
            uv_select = loss < loss.mean() + loss.std()
            if uv_select.sum() < 13:
                break
            uv_select = uv_select[:, np.newaxis]
    if poly is not None and try_poly:
        poly = find_1Dproj(poly[0]) / size
        sol = minimize(align_poly, np.array([0, 0, 0.6]), method='SLSQP', bounds=bounds, args=(poly, vertex, calib, size))
        if sol.success:
            t2 = sol.x
            d = distance(t, t2)
            if d > poly_protect[0]:
                t = t2
            elif d > poly_protect[1]:
                t = t * (1 - (d - poly_protect[1]) / (poly_protect[0] - poly_protect[1])) + t2 * ((d - poly_protect[1]) / (poly_protect[0] - poly_protect[1]))
    return vertex + t, success

def distance(x, y):
    return np.sqrt(((x - y)**2).sum())

def find_1Dproj(points):
    angles = [(0, 90), (-15, 75), (-30, 60), (-45, 45), (-60, 30), (-75, 15)]
    axs = [(np.array([[np.cos(x/180*np.pi), np.sin(x/180*np.pi)]]), np.array([np.cos(y/180*np.pi), np.sin(y/180*np.pi)])) for x, y in angles]
    proj = []
    for ax in axs:
        x = (points * ax[0]).sum(axis=1)
        y = (points * ax[1]).sum(axis=1)
        proj.append([x.min(), x.max(), y.min(), y.max()])
    return np.array(proj)

def align_poly(t, poly, vertex, calib, size):
    proj = perspective_np((vertex + t), calib)[:, :2]
    proj = find_1Dproj(proj) / size
    loss = (proj - poly)**2
    return loss.mean()

def align_uv(t, uv, vertex2xyz, calib):
    xyz = vertex2xyz + t
    proj = perspective_np(xyz, calib)[:, :2]
    loss = (proj - uv)**2
    return loss.mean()

#======================下面来自之前的trainer===========================

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), 'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.max(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

def get_final_preds(batch_heatmaps):
    #输入热图，输出关键点坐标
    coords, maxvals = get_max_preds(batch_heatmaps)
    # print(coords.shape)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                diff = np.array([hm[py][px + 1] - hm[py][px - 1],
                                 hm[py + 1][px] - hm[py - 1][px]])
                coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()
    # print(preds.shape)

    kp_point = np.abs(preds)
    return kp_point, maxvals


def normalization(img):
    img_channel = img.shape[0]
    img_f = torch.flatten(img, 1)
    min_, _ = torch.min(img_f, dim=-1)
    max_, _ = torch.max(img_f, dim=-1)
    min_ex = min_.reshape(img_channel, 1, 1).repeat((1,) + img.shape[-2:])
    max_ex = max_.reshape(img_channel, 1, 1).repeat((1,) + img.shape[-2:])
    img_n = (img - min_ex) / (max_ex - min_ex)
    return img_n

    # elif img.size() == 4:  # for img list
    #     batch_size = img.shape[0]
    #     img_channel = img.shape[1]
    #     for i in range(batch_size):  # batch_size
    #         img_f = torch.flatten(img[i, :, :, :], 1)
    #         min_, _ = torch.min(img_f, dim=-1)
    #         max_, _ = torch.max(img_f, dim=-1)
    #         min_ex = min_.reshape(img_channel, 1, 1).repeat((1,) + img.shape[-2:])
    #         max_ex = max_.reshape(img_channel, 1, 1).repeat((1,) + img.shape[-2:])
    #         img_n = (img - min_ex) / (max_ex - min_ex)
    #         return img_n


def load_pretrained_params(state_load, state_cur):
    state_dict = {k.replace('module.', ''): v for k, v in
                  state_load.items()}  # 如果是多卡训练的，参数名字里会有“module”前缀，单卡或者cpu测试要把前缀去掉
    pretrained_dict = {k: v for k, v in state_dict.items() if
                       k in state_cur and v.size() == state_cur[k].size()}  # 同时存在于当前模型（实例）和预训练模型中的参数

    wasted_module = [(k, v.size()) for k, v in state_dict.items() if
                     k not in state_cur or v.size() != state_cur[k].size()]  # 找出重载模型中有，而当前模型中没有的，或大小不同的模块
    print("wasted_module: {}".format(wasted_module))  # 打印出来看哪些参数没有load进去

    missing_modlue = [(k, v.size()) for k, v in state_cur.items() if
                      k not in state_dict or v.size() != state_dict[k].size()]  # 找出当前模型中存在,重载模型不存在的模块
    print("missing_modlue: {}".format(missing_modlue))
    state_cur.update(pretrained_dict)  # 更新要读取的参数 为 模型(实例) 和 模型参数能对应上的 参数
    # self.model.load_state_dict(state_cur)  # 读取模型
    # if self.mode == "train":
    #     print("load:{}".format([(k, v.size()) for k, v in state_cur.items()]))
    return state_cur

def vis_pose(kpt: np.ndarray, img=None):
    '''
    Input:
    :param kpt: keypoint array with shape [21,2]
    :param img: image
    :return:
    '''
    color_lst = ["yellow", "blue", "green", "cyan", "magenta"]
    group_lst = [1, 5, 9, 13, 17, 21]
    plt.figure()
    if img is not None:
        plt.imshow(img)
    for j, color in enumerate(color_lst):
        x = np.insert(kpt[group_lst[j]:group_lst[j + 1], 0], 0, kpt[0, 0])
        y = np.insert(kpt[group_lst[j]:group_lst[j + 1], 1], 0, kpt[0, 1])
        plt.plot(x, y, color=color, linewidth=3, marker=".", markerfacecolor="red", markersize=15, markeredgecolor="red")
    plt.axis('off')
    # plt.xlim((0, 200))
    # plt.ylim((200, 0))
    canvas = FigureCanvasAgg(plt.gcf())
    # 绘制图像
    canvas.draw()
    # 解码string 得到argb图像
    buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
    return buf

def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK0.5,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]]) # th=0.5
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred

def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists

def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1

def pck_at_threshold(dists, thr):
    '''
    为单个阈值计算 PCK。
    dists: 包含距离的 numpy 数组。
    thr: 阈值。
    '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1

def calculate_auc(dists, threshold_range=np.linspace(0, 0.5, 51)):
    '''
    计算 PCK 的曲线下面积 (AUC)。
    dists: 包含所有关键点距离的 numpy 数组。
    threshold_range: 用于评估 PCK 的一系列阈值。
    '''
    pck_values = [pck_at_threshold(dists, thr) for thr in threshold_range]
    pck_values = [p for p in pck_values if p != -1] # 过滤掉无效值

    if not pck_values:
        return 0.0

    # 使用梯形法则计算 AUC
    auc = np.trapz(pck_values, threshold_range[:len(pck_values)])
    
    # 将 AUC 归一化到 0 和 1 之间
    max_possible_auc = threshold_range[-1] - threshold_range[0]
    if max_possible_auc > 0:
        return auc / max_possible_auc
    return 0.0

class AUCMeter(object):
    """计算并存储距离，用于在周期结束时计算 AUC"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.dists = []

    def update(self, dists):
        self.dists.extend(dists)

    def get_epoch_auc(self):
        if not self.dists:
            return 0.0
        # 将所有距离的列表转换为一个扁平的 numpy 数组以进行计算
        all_dists = np.concatenate(self.dists)
        return calculate_auc(all_dists)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

class MPJPEMeter(object):
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def calculate_iou(pred, target, threshold=0.5):
    """
    为一批分割掩码计算交并比 (IoU).
    
    参数:
        pred (torch.Tensor): 预测的掩码，形状为 (B, C, H, W).
        target (torch.Tensor): 真实的掩码，形状为 (B, C, H, W).
        threshold (float): 用于二值化预测掩码的阈值.

    返回:
        float: 该批数据的平均 IoU 分数.
    """
    # 二值化预测结果
    pred = (pred > threshold).float()

    # 确保目标掩码也是二值化的 (0 或 1)
    target = (target > 0).float()

    # 计算交集和并集
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target).sum(dim=(1, 2, 3)) - intersection

    # 处理并集为零的特殊情况 (即预测和目标都为空)
    # 在这种情况下，如果交集也为0，则IoU为1，否则为0.
    iou = torch.where(union == 0, torch.tensor(1.0, device=pred.device), intersection / (union + 1e-8))
    
    return iou.mean().item()

class VertexMeter(object):
    """计算vertex相关指标的Meter"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.verts_errors = []
        self.pa_verts_errors = []
        self.f_scores_5mm = []
        self.f_scores_15mm = []
        self.pa_f_scores_5mm = []
        self.pa_f_scores_15mm = []

    def update(self, verts_error, pa_verts_error, f_score_5mm, f_score_15mm, pa_f_score_5mm=None, pa_f_score_15mm=None):
        self.verts_errors.extend(verts_error)
        self.pa_verts_errors.extend(pa_verts_error)
        self.f_scores_5mm.append(f_score_5mm)
        self.f_scores_15mm.append(f_score_15mm)
        if pa_f_score_5mm is not None:
            self.pa_f_scores_5mm.append(pa_f_score_5mm)
        if pa_f_score_15mm is not None:
            self.pa_f_scores_15mm.append(pa_f_score_15mm)

    def get_metrics(self):
        if not self.verts_errors:
            return None
        metrics = {
            'mpvpe': np.mean(self.verts_errors),
            'pa_mpvpe': np.mean(self.pa_verts_errors),
            'f_score_5mm': np.mean(self.f_scores_5mm),
            'f_score_15mm': np.mean(self.f_scores_15mm)
        }
        if self.pa_f_scores_5mm:
            metrics['pa_f_score_5mm'] = np.mean(self.pa_f_scores_5mm)
        if self.pa_f_scores_15mm:
            metrics['pa_f_score_15mm'] = np.mean(self.pa_f_scores_15mm)
        return metrics

def rigid_align_vertices(pred_vertices, gt_vertices):
    """
    对vertex进行刚体对齐
    Args:
        pred_vertices: [N, 3] 预测的vertices
        gt_vertices: [N, 3] ground truth vertices
    Returns:
        aligned_pred_vertices: [N, 3] 对齐后的预测vertices
    """
    return rigid_align(pred_vertices, gt_vertices)

def extract_joint_from_vertex(vertices, j_regressor):
    """
    从vertex提取关键点
    Args:
        vertices: [B, 778, 3] 或 [778, 3]
        j_regressor: [21, 778] 关节回归器
    Returns:
        joints: [B, 21, 3] 或 [21, 3]
    """
    if len(vertices.shape) == 3:  # batch dimension
        batch_size = vertices.shape[0]
        joints = []
        for i in range(batch_size):
            joint = np.matmul(j_regressor, vertices[i])
            joints.append(joint)
        return np.stack(joints, 0)
    else:  # single sample
        return np.matmul(j_regressor, vertices)
    

