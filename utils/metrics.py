# Originally written by wkentaro, additions by monsieurwave
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
import cv2
from sklearn.metrics import roc_curve, auc


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist


def segmentation_scores(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return {'overall_acc': acc,
            'mean_acc': acc_cls,
            'freq_w_acc': fwavacc,
            'mean_iou': mean_iu}


def dice_score_list(label_gt, label_pred, n_class):
    """

    :param label_gt: [WxH] (2D images)
    :param label_pred: [WxH] (2D images)
    :param n_class: number of label classes
    :return:
    """
    epsilon = 1.0e-6
    assert len(label_gt) == len(label_pred)
    batchSize = len(label_gt)
    dice_scores = np.zeros((batchSize, n_class), dtype=np.float32)
    for batch_id, (l_gt, l_pred) in enumerate(zip(label_gt, label_pred)):
        for class_id in range(n_class):
            img_A = np.array(l_gt == class_id, dtype=np.float32).flatten()
            img_B = np.array(l_pred == class_id, dtype=np.float32).flatten()
            score = 2.0 * np.sum(img_A * img_B) / (np.sum(img_A) + np.sum(img_B) + epsilon)
            dice_scores[batch_id, class_id] = score

    return np.mean(dice_scores, axis=0)


def dice_score(label_gt, label_pred, n_class):
    """

    :param label_gt:
    :param label_pred:
    :param n_class:
    :return:
    """

    epsilon = 1.0e-6
    assert np.all(label_gt.shape == label_pred.shape)
    dice_scores = np.zeros(n_class, dtype=np.float32)
    for class_id in range(n_class):
        img_A = np.array(label_gt == class_id, dtype=np.float32).flatten()
        img_B = np.array(label_pred == class_id, dtype=np.float32).flatten()
        score = 2.0 * np.sum(img_A * img_B) / (np.sum(img_A) + np.sum(img_B) + epsilon)
        dice_scores[class_id] = score

    return dice_scores


def roc_auc(label_gt, label_pred):
    y_true = np.array(label_gt).flatten()
    y_scores = np.array(label_pred).flatten()

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc_score = auc(fpr, tpr)
    return roc_auc_score


def single_class_dice_score(target, input):
    smooth = 1e-7
    iflat = np.array(input).flatten()
    tflat = np.array(target).flatten()
    intersection = (iflat * tflat).sum()

    return ((2. * intersection + smooth) /
            (iflat.sum() + tflat.sum() + smooth))


def precision_and_recall(label_gt, label_pred, n_class):
    from sklearn.metrics import precision_score, recall_score
    assert len(label_gt) == len(label_pred)
    precision = np.zeros(n_class, dtype=np.float32)
    recall = np.zeros(n_class, dtype=np.float32)
    img_A = np.array(label_gt, dtype=np.float32).flatten()
    img_B = np.array(label_pred, dtype=np.float32).flatten()
    precision[:] = precision_score(img_A, img_B, average=None, labels=range(n_class))
    recall[:] = recall_score(img_A, img_B, average=None, labels=range(n_class))

    return precision, recall


def distance_metric(seg_A, seg_B, dx, k):
    """
        Measure the distance errors between the contours of two segmentations.
        The manual contours are drawn on 2D slices.
        We calculate contour to contour distance for each slice.
        """

    # Extract the label k from the segmentation maps to generate binary maps
    seg_A = (seg_A == k)
    seg_B = (seg_B == k)

    table_md = []
    table_hd = []
    X, Y, Z = seg_A.shape
    for z in range(Z):
        # Binary mask at this slice
        slice_A = seg_A[:, :, z].astype(np.uint8)
        slice_B = seg_B[:, :, z].astype(np.uint8)

        # The distance is defined only when both contours exist on this slice
        if np.sum(slice_A) > 0 and np.sum(slice_B) > 0:
            # Find contours and retrieve all the points
            _, contours, _ = cv2.findContours(cv2.inRange(slice_A, 1, 1),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
            pts_A = contours[0]
            for i in range(1, len(contours)):
                pts_A = np.vstack((pts_A, contours[i]))

            _, contours, _ = cv2.findContours(cv2.inRange(slice_B, 1, 1),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
            pts_B = contours[0]
            for i in range(1, len(contours)):
                pts_B = np.vstack((pts_B, contours[i]))

            # Distance matrix between point sets
            M = np.zeros((len(pts_A), len(pts_B)))
            for i in range(len(pts_A)):
                for j in range(len(pts_B)):
                    M[i, j] = np.linalg.norm(pts_A[i, 0] - pts_B[j, 0])

            # Mean distance and hausdorff distance
            md = 0.5 * (np.mean(np.min(M, axis=0)) + np.mean(np.min(M, axis=1))) * dx
            hd = np.max([np.max(np.min(M, axis=0)), np.max(np.min(M, axis=1))]) * dx
            table_md += [md]
            table_hd += [hd]

    # Return the mean distance and Hausdorff distance across 2D slices
    mean_md = np.mean(table_md) if table_md else None
    mean_hd = np.mean(table_hd) if table_hd else None
    return mean_md, mean_hd

def Weighted_Binary_Cross_Entropy(gts, preds, n_class):
    """

    :param label_gt:
    :param label_pred:
    :param n_class:
    :return:
    """
    label_gt = gts[1]
    label_pred = preds[1]

    epsilon = 1.0e-6
    assert np.all(label_gt.shape == label_pred.shape)
    WBCE_score = 0.0
    
    img_gt = np.array(label_gt == 1, dtype=np.float32).flatten()
    img_pred = np.array(label_pred == 1, dtype=np.float32).flatten()
    
    N_plus = np.sum(img_gt)
    N_minus = np.sum(1-img_gt)
    R0 = N_plus / ( N_minus + N_plus )
    R1 = 1.0 - R0
    N = N_plus + N_minus
    img_pred_prob = epsilon+(1.0-2*epsilon)*img_pred
    WBCE_score = -(1.0/N) * np.sum(R0*img_gt*np.log(img_pred_prob) + R1*(1-img_gt)*np.log(1.0-img_pred_prob))

    return WBCE_score

def L1(gts, preds, n_class):
    """

    :param label_gt:
    :param label_pred:
    :param n_class:
    :return:
    """
    label_gt = gts[1]
    label_pred = preds[1]
    
    assert np.all(label_gt.shape == label_pred.shape)
    L1_score = 0.0
    
    img_gt = np.array(label_gt == 1, dtype=np.float32).flatten()
    img_pred = np.array(label_pred == 1, dtype=np.float32).flatten()
    L1_score = np.mean(np.abs(img_gt - img_pred))
    
    return L1_score

def VolumeL(gts, preds, n_class):
    """

    :param label_gt:
    :param label_pred:
    :param n_class:
    :return:
    """
    label_gt = gts[1]
    label_pred = preds[1]
    
    assert np.all(label_gt.shape == label_pred.shape)
    vol_score = 0.0
    
    img_gt = np.array(label_gt == 1, dtype=np.float32).flatten()
    img_pred = np.array(label_pred == 1, dtype=np.float32).flatten()
    N_plus = np.sum(img_gt)
    
    if N_plus!=0:
        vol_score = np.abs(np.sum(img_gt - img_pred)/N_plus)
    else:
        vol_score = np.abs(np.sum(img_gt - img_pred))
    return vol_score