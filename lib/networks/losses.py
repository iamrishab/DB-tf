import tensorflow as tf
from db_config import cfg


def dice_coefficient_loss(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-6
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    return loss


def balance_cross_entropy_loss(gt, pred, mask,
                               negative_ratio=3.0, eps=1e-6):
    positive = gt * mask
    negative = (1 - gt) * mask
    positive_count = tf.reduce_sum(positive)
    negative_count = tf.minimum(tf.reduce_sum(negative), positive_count * negative_ratio)
    negative_count = tf.cast(negative_count, tf.int32)
    gt = tf.reshape(gt, [-1, 1])
    pred = tf.reshape(pred, [-1, 1])
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt, logits=pred)
    positive_loss = cross_entropy * positive
    negative_loss = cross_entropy * negative
    negative_loss, _ = tf.nn.top_k(tf.reshape(negative_loss, [-1]), negative_count)

    negative_count = tf.cast(negative_count, tf.float32)
    balance_loss = (tf.reduce_sum(positive_loss) + tf.reduce_sum(negative_loss)) / (positive_count + negative_count + eps)

    return balance_loss

def softmax_cross_entropy_loss(y_true_cls, y_pred_cls, training_mask):
    '''
    softmax_cross_entropy(SCE) loss
    :param y_true_cls:[bs,w,h,N]
    :param y_pred_cls:[bs,w,h,N]
    :param training_mask:
    :return:
    '''
    re_mask = 1 - training_mask
    zero_mask = tf.zeros(tf.shape(re_mask))
    add_mask = tf.concat((re_mask, zero_mask, zero_mask), axis=3)

    y_true_cls = y_true_cls * training_mask + add_mask
    y_pred_cls = y_pred_cls * training_mask + add_mask

    y_true_cls = tf.reshape(y_true_cls, [-1, tf.shape(y_true_cls)[-1]])
    y_pred_cls = tf.reshape(y_pred_cls, [-1, tf.shape(y_true_cls)[-1]])

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true_cls, logits=y_pred_cls)
    cls_loss = tf.reduce_mean(cross_entropy)

    return cls_loss

def l1_loss(pred, gt, mask):

    loss = tf.reduce_mean(tf.abs(pred - gt) * mask) + 1e-6

    return loss


def smooth_l1_loss(pred, gt, mask, sigma=1.0):
    '''

    :param pred:
    :param gt: shape is same as pred
    :param sigma:
    :return:
    '''
    sigma2 = sigma**2

    diff = pred * mask - gt

    with tf.name_scope('smooth_l1_loss'):
        deltas_abs = tf.abs(diff)
        smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0 / sigma2), tf.float32)
        return tf.reduce_mean(tf.square(diff) * 0.5 * sigma2 * smoothL1_sign + \
               (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1))

def compute_cls_acc(pred, gt, mask):

    zero = tf.zeros_like(pred, tf.float32)
    one = tf.ones_like(pred, tf.float32)

    pred = tf.where(pred < 0.3, x=zero, y=one)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred * mask, gt * mask), tf.float32))

    return acc


def compute_loss(binarize_map, threshold_map, thresh_binary,
                 gt_score_maps, gt_threshold_map, gt_score_mask, gt_thresh_mask):

    binarize_loss = dice_coefficient_loss(gt_score_maps, binarize_map, gt_score_mask)
    threshold_loss = l1_loss(threshold_map, gt_threshold_map, gt_thresh_mask)
    thresh_binary_loss = dice_coefficient_loss(gt_score_maps, thresh_binary, gt_score_mask)

    model_loss = cfg.TRAIN.LOSS_ALPHA * binarize_loss + cfg.TRAIN.LOSS_BETA * threshold_loss + thresh_binary_loss

    tf.summary.scalar('losses/binarize_loss', binarize_loss)
    tf.summary.scalar('losses/threshold_loss', threshold_loss)
    tf.summary.scalar('losses/thresh_binary_loss', thresh_binary_loss)
    return model_loss

def compute_acc(binarize_map, threshold_map, thresh_binary,
                 gt_score_maps, gt_threshold_map, gt_score_mask, gt_thresh_mask):
    binarize_acc = compute_cls_acc(binarize_map, gt_score_maps, gt_score_mask)
    thresh_binary_acc = compute_cls_acc(threshold_map, gt_threshold_map, gt_thresh_mask)

    tf.summary.scalar('acc/binarize_acc', binarize_acc)
    tf.summary.scalar('acc/thresh_binary_acc', thresh_binary_acc)

    return binarize_acc, thresh_binary_acc



