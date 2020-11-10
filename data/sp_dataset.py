#python3 train.py  --dataroot ./datasets/datasets/data-six/ --name endtoend_no_shadow_ --checkpoints_dir all_checkpoints/endtoend_checkpoints/ --model pix2pix_endtoend_3 --dataset_mode endtoend_no_shadow --input_nc 1 --output_nc 1  --crop_size 256     --gpu_id 1 --netG unet_256 
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image

import json

class SpDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.C_paths=[]
        self.G_paths=[]
        self.H_paths=[]
        self.M_paths=[]
        self.S_paths=[]
        self.K_paths=[]
        self.Sh_paths=[]
        
        self.dir_AB = opt.dataroot  # get the image directory
        for i in os.listdir(self.dir_AB):
          T_path = self.dir_AB +'/'+ i+'/params/meta.ndjson';
          with open(T_path) as f:
            for line in f:
              j_content = json.loads(line)
              texture=j_content["texture"]["name"]
          if texture!='sha' and texture!='som':
            self.C_paths += make_dataset(self.dir_AB+'/'+i+'/r_contour/', opt.max_dataset_size)  # get image paths
            self.G_paths += make_dataset(self.dir_AB+'/'+i+'/r_gnomon/', opt.max_dataset_size)  # get image paths
            self.H_paths += make_dataset(self.dir_AB+'/'+i+'/r_highlights/', opt.max_dataset_size)  # get image paths
            self.M_paths += make_dataset(self.dir_AB+'/'+i+'/r_midtones/', opt.max_dataset_size)  # get image paths
            self.S_paths += make_dataset(self.dir_AB+'/'+i+'/r_shades/', opt.max_dataset_size)  # get image paths
            self.Sh_paths += make_dataset(self.dir_AB+'/'+i+'/r_shadow/', opt.max_dataset_size)  # get image paths
            self.K_paths += make_dataset(self.dir_AB+'/'+i+'/r_sketch/', opt.max_dataset_size)  # get image paths
        self.C_paths=sorted(self.C_paths)
        self.G_paths=sorted(self.G_paths)
        self.H_paths=sorted(self.H_paths)
        self.M_paths=sorted(self.M_paths)
        self.S_paths=sorted(self.S_paths)
        self.K_paths=sorted(self.K_paths)
        
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.input_nc 
        self.output_nc = self.opt.output_nc 

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        
        C_path = self.C_paths[index]
        T_path = self.dir_AB +'/'+ C_path.split('/')[-3]+'/params/meta.ndjson';
        G_path = self.G_paths[index]
        H_path = self.H_paths[index]
        M_path = self.M_paths[index]
        S_path = self.S_paths[index]
        Sh_path = self.Sh_paths[index]
        K_path = self.K_paths[index]
        AB_path=C_path
        try:
          A = Image.open(C_path).convert('RGB')
          B = Image.open(G_path).convert('RGB')
          G = Image.open(H_path).convert('RGB')
          H = Image.open(M_path).convert('RGB')
          I = Image.open(S_path).convert('RGB')
          J = Image.open(Sh_path).convert('RGB')
          K = Image.open(K_path).convert('RGB')
           
        except:
          print("EXCEPTION:"+str(AB_path))
        texture="zero"
        with open(T_path) as f:
            for line in f:
              j_content = json.loads(line)
              texture=j_content["texture"]["name"]
       
        C=Image.open("/nfs/151/gpu/raghav/data/shadegan/brushes_v2/"+texture+"/high.png").convert('RGB')
        
        D=Image.open("/nfs/151/gpu/raghav/data/shadegan/brushes_v2/"+texture+"/mid.png").convert('RGB')
        
        E=Image.open("/nfs/151/gpu/raghav/data/shadegan/brushes_v2/"+texture+"/shade.png").convert('RGB')
        
        F=Image.open("/nfs/151/gpu/raghav/data/shadegan/brushes_v2/"+texture+"/shadow.png").convert('RGB')
    
            # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        C_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        D_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        E_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        F_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        G_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        H_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        I_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        J_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        K_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)
        C = C_transform(C)
        D = D_transform(D)
        E = E_transform(E)
        F = F_transform(F)
        G = G_transform(G)
        H = H_transform(H)
        I = I_transform(I)
        J = J_transform(J)
        K = K_transform(K)

        return {'A': A, 'B': B,'C':C,'D':D, 'E':E, 'F':F, 'G':G,'H':H,'I':I,'J':J,'K':K,'A_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.C_paths)
