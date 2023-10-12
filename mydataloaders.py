import numpy as np


from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)


class myloadimg:
    '''input(img_bgr,imgsize=640,stride=23,auto=true)\n
    return img_pre,img_ori'''
    def __init__(self, img_bgr, img_size=640, stride=32, auto=True):
        

        
        self.img_bgr=img_bgr
        self.img_size = img_size
        self.stride = stride
        self.nf = 1
        self.video_flag = 1
        self.mode = 'image'
        self.auto = auto
        
    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        
        if self.count == 1:
            raise StopIteration
        self.count+=1
        
        # Read image

        img0 = self.img_bgr  # BGR
        
        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return  img, img0

