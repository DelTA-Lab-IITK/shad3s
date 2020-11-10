import torch
from .base_model import BaseModel
from . import networks
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance
import numpy as np;

class SeModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_SE_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss');parser.add_argument('--lambda_L2', type=float, default=100.0, help='weight for L1 loss');

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake'];
        self.visual_names = ['realInput_A','realInput_B','realInput_C','realInput_D','realInput_E','realInput_F','realOutput_A','realOutput_B','realOutput_C','realOutput_D','realOutput_E','fake_B_1','fake_B_2','fake_B_3','fake_B_4','fake_C'];
        
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D','G_2','D_2']
        else:  # during test time, only load G
            self.model_names = ['G','G_2']
            
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc*2, opt.output_nc*4, opt.ngf, opt.netG, 0,opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids);
        self.netG_2 = networks.define_G(opt.input_nc*8, opt.output_nc*1, opt.ngf, opt.netG,0, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc*2 + opt.output_nc*4, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids);
            self.netD_2 = networks.define_D(opt.input_nc*8 + opt.output_nc*1, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionBCE=torch.nn.BCELoss();self.Sigmoid=torch.nn.Sigmoid();
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999));self.optimizer_G_2 = torch.optim.Adam(self.netG_2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999));self.optimizer_D_2 = torch.optim.Adam(self.netD_2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G);self.optimizers.append(self.optimizer_G_2)
            self.optimizers.append(self.optimizer_D);self.optimizers.append(self.optimizer_D_2)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.realInput_A = input['A'].to(self.device) ;
        self.realInput_B = input['B'].to(self.device);
        self.realInput_C = input['C'].to(self.device);
        self.realInput_D = input['D'].to(self.device);
        self.realInput_E = input['E'].to(self.device);
        self.realInput_F = input['F'].to(self.device);
        self.realOutput_A = input['G'].to(self.device);
        self.realOutput_B = input['H'].to(self.device)#TODO
        self.realOutput_C = input['I'].to(self.device)#TODO
        self.realOutput_D = input['J'].to(self.device);
        self.realOutput_E = input['K'].to(self.device)
        self.real_A=torch.cat((self.realInput_A, self.realInput_B), 1);
        self.real_B=torch.cat((self.realOutput_A, self.realOutput_B), 1);
        self.real_B=torch.cat((self.real_B, self.realOutput_C), 1);
        self.real_B=torch.cat((self.real_B, self.realOutput_D), 1);
        self.image_paths = input['A_paths'];
        self.texture=torch.cat((self.realInput_C, self.realInput_D), 1);
        self.texture=torch.cat((self.texture, self.realInput_E), 1);
        self.texture=torch.cat((self.texture, self.realInput_F), 1);
        

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  ;
        C=torch.split(self.fake_B,1,1);
        self.fake_B_1=C[0];
        self.fake_B_2=C[1];
        self.fake_B_3=C[2];
        self.fake_B_4=C[3];
        
        self.real_C=torch.cat((self.fake_B, self.texture), 1);
        self.fake_C=self.netG_2(self.real_C);

    def backward_D(self):
        # Split model stage-I GAN loss for discriminator
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real)  * 0.5

        # Split model stage-II GAN loss for discriminator
        # Fake; stop backprop to the generator by detaching fake_C
        fake_AB_2 = torch.cat((self.real_C, self.fake_C), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake_2 = self.netD_2(fake_AB_2.detach())
        self.loss_D_fake_2 = self.criterionGAN(pred_fake_2, False)
        # Real
        real_AB_2 = torch.cat((self.real_C, self.realOutput_E), 1)
        pred_real_2 = self.netD_2(real_AB_2)
        self.loss_D_real_2 = self.criterionGAN(pred_real_2, True)
        # combine loss and calculate gradients
        self.loss_D_2 = (self.loss_D_fake_2 + self.loss_D_real_2) * 0.5
        
        # Total loss
        loss=self.loss_D+self.loss_D_2;
        loss.backward(retain_graph=True);

    def backward_G(self):
        # Split mosel stage-I GAN and L1 loss for the generator stage-I
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, (self.real_B) ) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
    
        # Split mosel stage-II GAN and L1 loss for the generator stage-II
        # First, G(A) should fake the discriminator
        fake_AB_2 = torch.cat((self.real_C, self.fake_C), 1)
        pred_fake_2 = self.netD_2(fake_AB_2)
        self.loss_G_GAN_2 = self.criterionGAN(pred_fake_2, True)
        # Second, G(A) = B
        self.loss_G_L1_2 = self.criterionL1(self.fake_C, self.realOutput_E) * self.opt.lambda_L2
        # combine loss and calculate gradients
        self.loss_G_2 = self.loss_G_GAN_2 + self.loss_G_L1_2
        
        # Total Loss
        loss=self.loss_G+self.loss_G_2;
        loss.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.optimizer_D_2.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        self.optimizer_D_2.step()          # update D's weights
        
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.optimizer_G_2.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step();
        self.optimizer_G_2.step();
        