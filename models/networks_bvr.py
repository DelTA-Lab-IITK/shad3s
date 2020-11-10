import torch
import torch.nn as nn
import functools

class UnetWithEmbeddingsGenerator(nn.Module):
    """Create a Unet-based generator with embeddings"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, nef=0, norm_layer=nn.BatchNorm2d, use_dropout=False,p=0.2):
        """Construct a UnetWithEmbeddings generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            nef (int)       -- the number of filters concatenated to embeddings
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetWithEmbeddingsGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetWithEmbeddingsSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, embed_nc=nef, submodule=None, norm_layer=norm_layer, innermost=True,p=p)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetWithEmbeddingsSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout,p=p)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetWithEmbeddingsSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,p=p)
        unet_block = UnetWithEmbeddingsSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer,p=p)
        unet_block = UnetWithEmbeddingsSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer,p=p)
        self.model = UnetWithEmbeddingsSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer,p=p)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetWithEmbeddingsSkipConnectionBlock(nn.Module):
    """Defines the UnetWithEmbeddings submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None, embed_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,p=0.2):
        """Construct a UnetWithEmbeddings submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            embed_nc (int) -- the number of channels concatenated to embeddings
            submodule (UnetWithEmbeddingsSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetWithEmbeddingsSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [UnetWithEmbeddingsDownWrapper(
                downconv
            )]
            up = [UnetWithEmbeddingsUpWrapper(
                nn.Sequential(
                    uprelu, upconv, nn.Tanh()
                )
            )]
            model = down + [submodule] + up
        elif innermost:
            fan_in = inner_nc
            if embed_nc :
                fan_in += embed_nc
            upconv = nn.ConvTranspose2d(fan_in, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)

            down = [UnetWithEmbeddingsDownWrapper(
                nn.Sequential(
                    downrelu, downconv
                )
            )]
            dup = [UnetWithEmbeddingsBottom()]
            up = [UnetWithEmbeddingsUpWrapper(
                nn.Sequential(
                    uprelu, upconv, upnorm
                )
            )]
            model = down + dup + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [UnetWithEmbeddingsDownWrapper(
                nn.Sequential(
                    downrelu, downconv, downnorm
                )
            )]
            up = [UnetWithEmbeddingsUpWrapper(
                nn.Sequential(
                    uprelu, upconv, upnorm
                )
            )]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(p)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        embeddings = None
        if isinstance(x, (list,tuple)) :
            x, embeddings = x
        if self.outermost:
            return self.model([x,embeddings])
        else:   # add skip connections
            y1_interm, y2_interm = self.model([x,embeddings])
            return y1_interm, torch.cat([x, y2_interm], 1)

class UnetWithEmbeddingsDownWrapper(nn.Module) :
    def __init__(self, down_net) :
        super(UnetWithEmbeddingsDownWrapper, self).__init__()
        self.down_net = down_net

    def forward(self, x) :
        x1, x2 = x
        return self.down_net(x1), x2

class UnetWithEmbeddingsUpWrapper(nn.Module) :
    def __init__(self, up_net) :
        super(UnetWithEmbeddingsUpWrapper, self).__init__()
        self.up_net = up_net

    def forward(self, x) :
        x1, x2 = x
        return x1, self.up_net(x2)

class UnetWithEmbeddingsBottom(nn.Module) :
    def forward(self, x) :
        x1, x2 = x
        x = torch.cat([x1,x2], 1) if x2 is not None else x1
        return x, x

def test() :
    class Identity(nn.Module):
        def __init__(self, a) :
            super(Identity, self).__init__()
        def forward(self, x):
            return x

    # 64p
    genet = UnetWithEmbeddingsGenerator(2, 3, 6, 64, norm_layer=Identity, use_dropout=False)
    print(genet)

    # NCHW
    x = torch.rand(8,2,64,64)
    y1, y2 = genet(x)
    print(
        tuple(y1.size()),
        tuple(y2.size()),
    )

    # 64p
    genet = UnetWithEmbeddingsGenerator(2, 3, 6, 64, 512, norm_layer=Identity, use_dropout=False)
    y1, y2 = genet([x, y1])
    print(
        tuple(y1.size()),
        tuple(y2.size()),
    )

if __name__ == '__main__' :

  test()
