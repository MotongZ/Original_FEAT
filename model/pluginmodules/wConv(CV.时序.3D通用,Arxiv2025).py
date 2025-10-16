import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple


class wConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, den, stride=1, padding=2, groups=1, dilation=1,
                 bias=False):
        super(wConv1d, self).__init__()
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.kernel_size = _single(kernel_size)
        self.groups = groups
        self.dilation = _single(dilation)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        device = torch.device('cpu')
        self.register_buffer('alfa', torch.cat([torch.tensor(den, device=device), torch.tensor([1.0], device=device),
                                                torch.flip(torch.tensor(den, device=device), dims=[0])]))
        self.register_buffer('Phi', self.alfa)

        if self.Phi.shape != self.kernel_size:
            raise ValueError(f"Phi shape {self.Phi.shape} must match kernel size {self.kernel_size}")

    def forward(self, x):
        Phi = self.Phi.to(x.device)
        weight_Phi = self.weight * Phi
        return F.conv1d(x, weight_Phi, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups,
                        dilation=self.dilation)


class wConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, den, stride=1, padding=2, groups=1, dilation=1,
                 bias=False):
        super(wConv2d, self).__init__()
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.kernel_size = _pair(kernel_size)
        self.groups = groups
        self.dilation = _pair(dilation)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        device = torch.device('cpu')
        self.register_buffer('alfa', torch.cat([torch.tensor(den, device=device), torch.tensor([1.0], device=device),
                                                torch.flip(torch.tensor(den, device=device), dims=[0])]))
        self.register_buffer('Phi', torch.outer(self.alfa, self.alfa))

        if self.Phi.shape != self.kernel_size:
            raise ValueError(f"Phi shape {self.Phi.shape} must match kernel size {self.kernel_size}")

    def forward(self, x):
        Phi = self.Phi.to(x.device)
        weight_Phi = self.weight * Phi
        return F.conv2d(x, weight_Phi, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups,
                        dilation=self.dilation)


class wConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, den, stride=1, padding=2, groups=1, dilation=1,
                 bias=False):
        super(wConv3d, self).__init__()
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.kernel_size = _triple(kernel_size)
        self.groups = groups
        self.dilation = _triple(dilation)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        device = torch.device('cpu')
        self.register_buffer('alfa', torch.cat([torch.tensor(den, device=device), torch.tensor([1.0], device=device),
                                                torch.flip(torch.tensor(den, device=device), dims=[0])]))
        self.register_buffer('Phi', torch.einsum('i,j,k->ijk', self.alfa, self.alfa, self.alfa))

        if self.Phi.shape != self.kernel_size:
            raise ValueError(f"Phi shape {self.Phi.shape} must match kernel size {self.kernel_size}")

    def forward(self, x):
        Phi = self.Phi.to(x.device)
        weight_Phi = self.weight * Phi
        return F.conv3d(x, weight_Phi, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups,
                        dilation=self.dilation)

if __name__ == '__main__':
    # Test wConv1d
    print("Testing wConv1d...")
    den = [0.5, 0.75]  # Example density values
    block = wConv1d(in_channels=3, out_channels=3, kernel_size=5, den=den).to('cuda')
    input = torch.rand(1, 3, 32).to('cuda')  # batch_size=1, channels=3, length=32
    output = block(input)
    print("Input size:", input.size())
    print("Output size:", output.size())
    print()

    # Test wConv2d
    print("Testing wConv2d...")
    den = [0.5, 0.75]  # Example density values
    block = wConv2d(in_channels=3, out_channels=3, kernel_size=5, den=den).to('cuda')
    input = torch.rand(1, 3, 32, 32).to('cuda')  # batch_size=1, channels=3, height=32, width=32
    output = block(input)
    print("Input size:", input.size())
    print("Output size:", output.size())
    print()

    # Test wConv3d
    print("Testing wConv3d...")
    den = [0.5, 0.75]  # Example density values
    block = wConv3d(in_channels=3, out_channels=3, kernel_size=5, den=den).to('cuda')
    input = torch.rand(1, 3, 32, 32, 32).to('cuda')  # batch_size=1, channels=3, depth=32, height=32, width=32
    output = block(input)
    print("Input size:", input.size())
    print("Output size:", output.size())