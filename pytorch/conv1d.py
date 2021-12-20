import torch
from torch import nn
import torch.nn.functional as F

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, momentum):
        super(ConvBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels, momentum)
        self.bn2 = nn.BatchNorm1d(out_channels, momentum)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=2, pool_type='avg'):
        """
        Args:
          input: (batch_size, in_channels, time_steps, freq_bins)

        Outputs:
          output: (batch_size, out_channels, classes_num)
        """

        x = F.relu_(self.bn1(self.conv1(input)))
        x = F.relu_(self.bn2(self.conv2(x)))

        if pool_type == 'avg':
            x = F.avg_pool1d(x, kernel_size=pool_size)

        return x


class ConvStack1D(nn.Module):
    def __init__(self, momentum):
        super(ConvStack1D, self).__init__()
        self.conv_block1 = ConvBlock1D(in_channels=1, out_channels=48, momentum=momentum)
        self.conv_block2 = ConvBlock1D(in_channels=48, out_channels=64, momentum=momentum)
        self.conv_block3 = ConvBlock1D(in_channels=64, out_channels=96, momentum=momentum)
        self.conv_block4 = ConvBlock1D(in_channels=96, out_channels=128, momentum=momentum)
        self.fc5 = nn.Linear(1024, 384, bias=False)
        self.bn5 = nn.BatchNorm1d(384, momentum=momentum)
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc5)
        init_bn(self.bn5)

    def forward(self, x):
        x = x.transpose(1, 2).contiguous().view(-1, 1, 128)
        x = self.conv_block1(x, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training).view(24, 625, 128, -1)
        x = x.flatten(2)
        x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x = F.dropout(x, p=0.5, training=self.training, inplace=True)
        return x
