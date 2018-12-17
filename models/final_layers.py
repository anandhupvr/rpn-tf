import tensorflow as tf

def roi_pool(net, rois, img):

    boxes = rois[:,0:]
    norm = tf.cast(tf.stack([img[1],img[0],img[1],img[0]],axis=0),dtype=tf.float32)
    boxes = tf.div(boxes, norm)
    boxes = tf.stack([boxes[:,1],boxes[:,0],boxes[:,3],boxes[:,2]],axis=1)
    net_NHWC = tf.transpose(net, [0, 2, 1, 3])
    resize = tf.image.resize_images(net_NHWC, (14, 14))
    proposed = tf.image.crop_to_bounding_box(net_NHWC, int(boxes[0][0]),int(boxes[0][1]),14,14)
    pooled = tf.layers.max_pooling2d(inputs=proposed, pool_size=[2, 2], strides=2)

    return pooled

def classification(fc7):

    classe = tf.layers.conv2d(fc7,
                            filters= 1,
                            kernel_size=(1, 1),
                            activation='sigmoid',
                            kernel_initializer='uniform',
                            name="rpn_out_class")
    classe = tf.nn.softmax(classe)
    
    return classe
def regression(fc7):

    reg = tf.layers.conv2d(fc7,
                            filters=4,
                            kernel_size=(1, 1),
                            activation='linear',
                            kernel_initializer='uniform',
                            name='rpn_out_regre')
    return reg
def flat(pooled):

    fc6 = tf.layers.conv2d(pooled, 4096, [7, 7], padding='VALID')
    fc7 = tf.layers.conv2d(fc6, 4096, [1, 1])
    label = classification(fc7)
    cored = regression(fc7)

    return label, cored