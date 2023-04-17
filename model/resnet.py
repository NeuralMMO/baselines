
import torch.nn as nn
import numpy as np

def same_padding(in_size, filter_size, stride_size):
  in_height, in_width = in_size
  if isinstance(filter_size, int):
    filter_height, filter_width = filter_size, filter_size
  else:
    filter_height, filter_width = filter_size
  stride_height, stride_width = stride_size

  out_height = np.ceil(float(in_height) / float(stride_height))
  out_width = np.ceil(float(in_width) / float(stride_width))

  pad_along_height = int(
    ((out_height - 1) * stride_height + filter_height - in_height))
  pad_along_width = int(
    ((out_width - 1) * stride_width + filter_width - in_width))
  pad_top = pad_along_height // 2
  pad_bottom = pad_along_height - pad_top
  pad_left = pad_along_width // 2
  pad_right = pad_along_width - pad_left
  padding = (pad_left, pad_right, pad_top, pad_bottom)
  output = (out_height, out_width)
  return padding, output

class SlimConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel, stride, padding,
               initializer="default", activation_fn=None, bias_init=0):
    super(SlimConv2d, self).__init__()
    layers = []

    # Padding layer.
    if padding:
      layers.append(nn.ZeroPad2d(padding))

    # Actual Conv2D layer (including correct initialization logic).
    conv = nn.Conv2d(in_channels, out_channels, kernel, stride)
    if initializer:
      if initializer == "default":
          initializer = nn.init.xavier_uniform_
      initializer(conv.weight)
    nn.init.constant_(conv.bias, bias_init)
    layers.append(conv)
    if activation_fn is not None:
      layers.append(activation_fn())

    # Put everything in sequence.
    self._model = nn.Sequential(*layers)

  def forward(self, x):
      return self._model(x)

class ResidualBlock(nn.Module):
  def __init__(self, i_channel, o_channel, in_size, kernel_size=3, stride=1):
    super().__init__()
    self._relu = nn.ReLU(inplace=True)

    padding, out_size = same_padding(in_size, kernel_size, [stride, stride])
    self._conv1 = SlimConv2d(
      i_channel, o_channel,
      kernel=3, stride=stride,
      padding=padding, activation_fn=None)

    padding, out_size = same_padding(out_size, kernel_size, [stride, stride])
    self._conv2 = SlimConv2d(o_channel, o_channel,
                              kernel=3, stride=stride,
                              padding=padding, activation_fn=None)

    self.padding, self.out_size = padding, out_size

  def forward(self, x):
    out = self._relu(x)
    out = self._conv1(out)
    out = self._relu(out)
    out = self._conv2(out)
    out += x
    return out


class ResNet(nn.Module):
  def __init__(self, in_ch, in_size, channel_and_blocks=None):
    super().__init__()

    out_size = in_size
    conv_layers = []
    if channel_and_blocks is None:
      channel_and_blocks = [(16, 2), (32, 2), (32, 2)]

    for (out_ch, num_blocks) in channel_and_blocks:
      # Downscale
      padding, out_size = same_padding(out_size, filter_size=3,
                                        stride_size=[1, 1])
      conv_layers.append(
        SlimConv2d(in_ch, out_ch, kernel=3, stride=1, padding=padding,
                    activation_fn=None))

      padding, out_size = same_padding(out_size, filter_size=3,
                                       stride_size=[2, 2])
      conv_layers.append(nn.ZeroPad2d(padding))
      conv_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

      # Residual blocks
      for _ in range(num_blocks):
        res = ResidualBlock(
          i_channel=out_ch, o_channel=out_ch, in_size=out_size)
        conv_layers.append(res)

      padding, out_size = res.padding, res.out_size
      in_ch = out_ch

    conv_layers.append(nn.ReLU(inplace=True))
    self.resnet = nn.Sequential(*conv_layers)

  def forward(self, x):
    return self.resnet(x)
