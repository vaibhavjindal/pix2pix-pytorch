import torch.utils.data as data
from PIL import Image,ImageOps
import os
import random 

class GetLoader(data.Dataset):
    def __init__(self,data_root,data_list,transform1=None,transform2=None):
        self.root = data_root
        self.transform1 = transform1
        self.transform2 = transform2

        f = open(data_list,'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []

        for data in data_list:
            self.img_paths.append(data[:-1])

    def __getitem__(self,item):
        img_paths = self.img_paths[item]
        imgs = Image.open(os.path.join(self.root,img_paths))
        width,height = imgs.size
        area1 = (0,0,width/2,height)
        area2 = (width/2,0,width,height)
        img1 = imgs.crop(area1)
        img2 = imgs.crop(area2)
        img1,img2 = resize_crop(img1,img2,286,256)
        img1,img2 = random_mirror(img1,img2,0.5)

        if self.transform1 is not None:
            img1 = self.transform1(img1)

        if self.transform2 is not None:
            img2 = self.transform2(img2)

        return img1,img2

    def __len__(self):
        return self.n_data

def resize_crop(img1,img2,size = 286,crop_size=256):
    img1 = img1.resize((size,size), Image.ANTIALIAS)
    img2 = img2.resize((size,size), Image.ANTIALIAS)
    x_rand = random.randint(0,size-1-crop_size)
    y_rand = random.randint(0,size-1-crop_size)
    area = (x_rand,y_rand,x_rand+crop_size,y_rand+crop_size)
    img1 = img1.crop(area)
    img2 = img1.crop(area)
    return img1,img2


def random_mirror(img1,img2,p=0.5):
    n=random.uniform(0,1)
    if n>p:
        img1 = ImageOps.mirror(img1)    
        img2 = ImageOps.mirror(img2)

    return img1,img2    

        