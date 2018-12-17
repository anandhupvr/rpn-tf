import tensorflow as tf





def rpn_cls_loss(rpn_cls_score,rpn_labels):
    '''
    Calculate the Region Proposal Network classifier loss. Measures how well 
    the RPN is able to propose regions by the performance of its "objectness" 
    classifier.
    
    Standard cross-entropy loss on logits
    '''
    with tf.variable_scope('rpn_cls_loss'):
        # input shape dimensions
        shape = tf.shape(rpn_cls_score)
        
        # Stack all classification scores into 2D matrix
        rpn_cls_score = tf.transpose(rpn_cls_score,[0,3,1,2])
        rpn_cls_score = tf.reshape(rpn_cls_score,[shape[0],2,shape[3]//2*shape[1],shape[2]])
        rpn_cls_score = tf.transpose(rpn_cls_score,[0,2,3,1])
        rpn_cls_score = tf.reshape(rpn_cls_score,[-1,2])
        
        # Stack labels
        rpn_labels = tf.reshape(rpn_labels,[-1])
        
        # Ignore label=-1 (Neither object nor background: IoU between 0.3 and 0.7)
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score,tf.where(tf.not_equal(rpn_labels,-1))),[-1,2])
        rpn_labels = tf.reshape(tf.gather(rpn_labels,tf.where(tf.not_equal(rpn_labels,-1))),[-1])
        
        # Cross entropy error
        rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_labels))
    
    return rpn_cross_entropy
    
    
def rpn_bbox_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_inside_weights, rpn_outside_weights):
    '''
    Calculate the Region Proposal Network bounding box loss. Measures how well 
    the RPN is able to propose regions by the performance of its localization.

    lam/N_reg * sum_i(p_i^* * L_reg(t_i,t_i^*))

    lam: classification vs bbox loss balance parameter     
    N_reg: Number of anchor locations (~2500)
    p_i^*: ground truth label for anchor (loss only for positive anchors)
    L_reg: smoothL1 loss
    t_i: Parameterized prediction of bounding box
    t_i^*: Parameterized ground truth of closest bounding box
    '''    
    with tf.variable_scope('rpn_bbox_loss'):
        # Transposing
        rpn_bbox_targets = tf.transpose(rpn_bbox_targets, [0,2,3,1])
        rpn_inside_weights = tf.transpose(rpn_inside_weights, [0,2,3,1])
        rpn_outside_weights = tf.transpose(rpn_outside_weights, [0,2,3,1])
        
        # How far off was the prediction?
        diff = tf.multiply(rpn_inside_weights, rpn_bbox_pred - rpn_bbox_targets)
        diff_sL1 = smoothL1(diff, 3.0)
        
        # Only count loss for positive anchors. Make sure it's a sum.
        rpn_bbox_reg = tf.reduce_sum(tf.multiply(rpn_outside_weights, diff_sL1))
    
        # Constant for weighting bounding box loss with classification loss
        rpn_bbox_reg = cfg.TRAIN.RPN_BBOX_LAMBDA * rpn_bbox_reg
    
    return rpn_bbox_reg    
    



    
