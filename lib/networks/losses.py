import tensorflow as tf

def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    tf.summary.scalar('classification_dice_loss', loss)
    return loss


def softmax_cross_entropy(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    softmax_cross_entropy(SCE) loss
    :param y_true_cls:[bs,w,h,3]
    :param y_pred_cls:[bs,w,h,3]
    :param training_mask:
    :return:
    '''
    re_mask = 1 - training_mask
    zero_mask = tf.zeros(tf.shape(re_mask))
    add_mask = tf.concat((re_mask, zero_mask, zero_mask), axis=3)

    y_true_cls = y_true_cls * training_mask + add_mask
    y_pred_cls = y_pred_cls * training_mask + add_mask

    y_true_cls = tf.reshape(y_true_cls, [-1, 3])
    y_pred_cls = tf.reshape(y_pred_cls, [-1, 3])

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true_cls, logits=y_pred_cls)
    cls_loss = tf.reduce_mean(cross_entropy)

    # tf.summary.scalar('classification_sce_loss', cls_loss)
    return cls_loss


def _smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=1.0):
    '''

    :param bbox_pred: [-1, 4] in RPN. [-1, cls_num+1, 4] or [-1, cls_num+1, 5] in Fast-rcnn
    :param bbox_targets: shape is same as bbox_pred
    :param sigma:
    :return:
    '''
    sigma_2 = sigma**2

    box_diff = bbox_pred - bbox_targets

    abs_box_diff = tf.abs(box_diff)

    smoothL1_sign = tf.stop_gradient(
        tf.to_float(tf.less(abs_box_diff, 1. / sigma_2)))
    loss_box = tf.pow(box_diff, 2) * (sigma_2 / 2.0) * smoothL1_sign \
               + (abs_box_diff - (0.5 / sigma_2)) * (1.0 - smoothL1_sign)
    return loss_box


def total_loss(y_true_cls, y_pred_cls,
         y_true_geo, y_pred_geo,
         training_mask):
    '''
    define the loss used for training, contraning two part,
    the first part we use dice loss instead of weighted logloss,
    the second part is the iou loss defined in the paper
    :param y_true_cls: ground truth of text
    :param y_pred_cls: prediction os text
    :param y_true_geo: ground truth of geometry
    :param y_pred_geo: prediction of geometry
    :param training_mask: mask used in training, to ignore some text annotated by ###
    :return:
    '''
    classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
    # scale classification loss to match the iou loss part
    classification_loss *= 0.01

    # d1 -> top, d2->right, d3->bottom, d4->left
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)

    # 计算真实旋转框、预测旋转框面积
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)

    # 计算相交部分的高度和宽度  面积
    w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
    h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)

    # 计算R_true与R_pred的交集
    area_intersect = w_union * h_union

    # 计算R_true与R_pred的并集
    area_union = area_gt + area_pred - area_intersect

    # IoU loss,加1为了防止交集为0，log0没意义
    L_AABB = -tf.log((area_intersect + 1.0) / (area_union + 1.0))

    # 夹角的loss
    L_theta = 1 - tf.cos(theta_pred - theta_gt)
    # tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls * training_mask))
    # tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls * training_mask))

    valid_pts_nums = tf.cast(tf.count_nonzero(y_true_cls * training_mask), dtype=tf.float32)
    tf.summary.scalar('geometry_AABB', tf.reduce_sum(L_AABB * y_true_cls * training_mask) / valid_pts_nums)
    tf.summary.scalar('geometry_theta', tf.reduce_sum(L_theta * y_true_cls * training_mask) / valid_pts_nums)

    L_g = L_AABB + 20 * L_theta  # geometry_map loss

    return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss
