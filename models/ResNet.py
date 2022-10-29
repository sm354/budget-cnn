import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Step_<module> functions:
    parameters = weight and bias
    conv.weight.shape = (planes,in_planes,filtersize,filtersize)
    bnorm.weight.shape = number of channels 
    bnorm step up : 4 -> 16 -> 64
    fc: fully connected linear layer

    using torch.no_grad() to have non-differentiable step [https://discuss.pytorch.org/t/leaf-variable-was-used-in-an-inplace-operation/308/2]
'''

def Step_Linear(nn_linear):
    num_class,dim=nn_linear.weight.shape 
    fc=nn.Linear(2*dim,num_class)
    for i in range(num_class):
        for j in range(dim):
            fc.weight[i, 2*j:2*j+2] = nn_linear.weight[i,j]
        fc.bias[i]=nn_linear.bias[i]
    return fc

def Step_Conv(nn_conv2d):
    '''
    takes in nn.Conv2d of f=3 and Steps it Up
    -stride is maintained 
    -pooling=1 (to maintain img same)
    -copy paste 3,3 to 4,3,3
    '''
    ouc,inc=nn_conv2d.weight.shape[:-2]
    c=nn.Conv2d(2*inc,2*ouc,kernel_size=3,stride=nn_conv2d.stride,padding=1,bias=False)
    for i in range(ouc):
        for j in range(inc):
            c.weight[2*i:2*i+2, 2*j:2*j+2, :, :]=nn_conv2d.weight[i,j,:,:]
    return c

def Step_BN(nn_bnorm2d):
    dim=nn_bnorm2d.weight.shape[0]
    b=nn.BatchNorm2d(2*dim)
    for i in range(dim):
        b.weight[2*i:2*i+2]=nn_bnorm2d.weight[i]
        b.bias[2*i:2*i+2]=nn_bnorm2d.bias[i]
    return b

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.inc=in_channels
        self.ouc=out_channels
        self.s=stride
        self.ks=kernel_size
        self.p=padding
        self.shortcut=nn.Sequential()

        ####Residual Block#####
        self.bn1=nn.BatchNorm2d(in_channels)
        self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.conv2=nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,stride=1,padding=padding,bias=False)
        if in_channels != out_channels: 
            self.shortcut=nn.Sequential(
                # nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels=self.inc,out_channels=self.ouc,kernel_size=1,stride=self.s,padding=0,bias=False), 
            )

    def forward(self, inp):
        out=F.relu(self.bn1(inp))
        shortcut=self.shortcut(out) if self.inc!=self.ouc else inp
        out=self.conv1(out)
        out=self.conv2(F.relu(self.bn2(out)))
        out+=shortcut
        return out

    def Step_RB(self):
        self.inc*=2
        self.ouc*=2

        with torch.no_grad():
            self.bn1=Step_BN(self.bn1)
            self.bn2=Step_BN(self.bn2)
            self.conv1=Step_Conv(self.conv1)
            self.conv2=Step_Conv(self.conv2)

            if self.inc!=self.ouc:
                # b=Step_BN(self.shortcut[0]) 
                c=nn.Conv2d(self.inc, self.ouc, kernel_size=1, stride=self.s, padding=0, bias=False)
                for j in range(self.inc//2):
                    for i in range(self.ouc//2):
                        # c.weight[2*i:2*i+2,2*j:2*j+2,:,:]=self.shortcut[1].weight[i,j,:,:]
                        c.weight[2*i:2*i+2,2*j:2*j+2,:,:]=self.shortcut[0].weight[i,j,:,:]
                # self.shortcut=nn.Sequential(b,c)
                self.shortcut=nn.Sequential(c)


class ResNet(nn.Module):
    def __init__(self, n=3, r=10, ks=3, io=64, img_channels=3): 
        # ks: kernel_size
        # io: initial out_channels determining model size
        super(ResNet, self).__init__()
        self.n=n
        self.r=r
        self.inc=io
        self.ks=ks

        self.conv1=nn.Conv2d(in_channels=img_channels,out_channels=io,kernel_size=ks,stride=1,padding=1,bias=False) #first conv layer
        
        #6n layers : each 2n layers of same feature map size
        blocks_A, blocks_B, blocks_C, blocks_D = [ResidualBlock(io,io,stride=1)],[ResidualBlock(io,2*io,stride=2)],[ResidualBlock(2*io,4*io,stride=2)],[ResidualBlock(4*io,8*io,stride=2)]
        for i in range(1,n):
            blocks_A.append(ResidualBlock(io,io,stride=1))
            blocks_B.append(ResidualBlock(2*io,2*io,stride=1))
            blocks_C.append(ResidualBlock(4*io,4*io,stride=1))
            blocks_D.append(ResidualBlock(8*io,8*io,stride=1))
        
        self.blocks_A = nn.Sequential(*blocks_A) #feature map size 32,32
        self.blocks_B = nn.Sequential(*blocks_B) #feature map size 16,16
        self.blocks_C = nn.Sequential(*blocks_C) #feature map size 8,8
        self.blocks_D = nn.Sequential(*blocks_D) #feature map size 4,4

        # self.GAP = nn.AvgPool2d(kernel_size=8) : removing this since for smaller images kernel size would be different
        self.fc = nn.Linear(in_features=8*io,out_features=r,bias=True)

    def forward(self, inp):
        out=self.conv1(inp)
        # out=self.bn1(out)
        # out=F.relu(out)
        out=self.blocks_A(out)
        out=self.blocks_B(out)
        out=self.blocks_C(out)
        out=self.blocks_D(out)
        # out=self.GAP(out)
        out = F.avg_pool2d(out, out.size(2))
        out=out.view(out.size(0),-1)
        out=self.fc(out)
        return out

    def Step_ResNet(self):
        self.inc*=2
        with torch.no_grad():
            img_c=self.conv1.in_channels
            c=nn.Conv2d(in_channels=img_c, out_channels=self.inc, kernel_size=self.ks, stride=1, padding=1, bias=False)
            for j in range(img_c):
                for i in range(self.inc//2):
                    # 4,3,3,3 -> 8,3,3,3 : 3,3 to 2,3,3 : copy paste
                    c.weight[2*i:2*i+2,j,:,:]=self.conv1.weight[i,j,:,:]
            self.conv1=c

            for i in range(self.n):
                self.blocks_A[i].Step_RB()
                self.blocks_B[i].Step_RB()
                self.blocks_C[i].Step_RB()
                self.blocks_D[i].Step_RB()

            self.fc=Step_Linear(self.fc)