


class Config:
    def __init__(self):
        self.network = 'vgg16'
        self.anchor_box_scales = [128, 256, 512]
        self.anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]
        self.vgg16_path = 'vgg_16_2016_08_28/vgg16.ckpt'
        self.anchor_scales = [8, 16, 32]
        self.image_width = 224
        self.image_height = 224
