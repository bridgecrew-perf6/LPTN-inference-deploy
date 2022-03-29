import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

from codes.models import create_model
from codes.utils.dist_util import get_dist_info, init_dist
from codes.utils import set_random_seed, tensor2img
from codes.utils.options import parse

import random
import requests
from io import BytesIO
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DummyArgs():
    '''
    A class which mimics the behavior of ArgParser()
    '''
    def __init__(self, yml_filepath):
        self.launcher = 'none' # as for test, it is the default setting
        self.local_rank = 0 # default setting for original parse_options() function
        self.opt = yml_filepath


def parse_options(args, is_train = True):
    '''
    Overriding the original code flow of this function because we do not want a script but want to call it inside a function
    Original function is present at codes/train.py
    '''
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt


def build_model(yml_filepath = './options/test/LPTN/test_FiveK.yml'):

    args = DummyArgs(yml_filepath) # This class mimics the behaviour of the data structure which -parse_options()- is expecting
    opt = parse_options(args, is_train=False)  # parse options, set distributed setting, set ramdom seed

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # create model
    model = create_model(opt)
    return model.net_g.eval()


def load_image(path:str, mean:[float,tuple] = None, std:[float,tuple] = None):
    '''
    Load the image given a link or a path and convert it to Pytorch Image Tensor. Normalize only and only if any of the Mean or STD are given
    args:
      path: image link or path
      mean: Mean of the distribution from where it was taken or mean of the training set: could be a single number or a tuple of 3 numbers for each channel (r,g,b)
      std: Standard deviation of the distribution: could be a single number or a tuple of 3 numbers for each channel (r,g,b)

    returns:
      An image of dimensions (C,H,W) with range in [0-1] and dtype as float32
    '''

    if ("https" in path) or ("http" in path):
      image = Image.open(BytesIO(requests.get(path).content))

    else:
      image = Image.open(path)

    if image.mode != 'RGB':
      image = image.convert('RGB')
    
    # Check for the comments at line 45,46, 78 inside ./data/paired_image_dataset. They are using a 2 stage operation
    tensor_image = TF.to_tensor(image).to(DEVICE) # Originally they are using 1. utils.img_utils.imfrombytes  -> 2. utils.img_utils.img2tensor

    # taken from __getitem__() from the PairedImageDataset / UnPairedImageDataset Class defined in the module ./data/
    if mean is not None or std is not None:
            TF.normalize(tensor_image, mean, std, inplace=True)

    return tensor_image


def evaluate(model, url,  plot: False):
    '''
    Get inference from the model
    pass in model object and the url of the 
    '''
    torch_image = load_image(url)

    with torch.no_grad():
        result = model(torch_image.unsqueeze(0)) 
        result = tensor2img(result.detach().cpu(), rgb2bgr = False) # Result is from -1 to 1. Need to rescale it from 0-1 or 0-255

    if plot:
        f, ax = plt.subplots(1,2, figsize = (20,10))
        ax[0].imshow(np.array(Image.open(BytesIO(requests.get(url).content))))
        ax[1].imshow(result)
        plt.show()

    return result
    
