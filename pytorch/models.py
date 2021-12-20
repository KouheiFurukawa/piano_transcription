import os
import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MuLawEncoding

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from pytorch_utils import move_data_to_device


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


def init_gru(rnn):
    """Initialize a GRU layer. """

    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)

        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in: (i + 1) * fan_in, :])

    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))

    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, momentum):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels, momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        """
        Args:
          input: (batch_size, in_channels, time_steps, freq_bins)

        Outputs:
          output: (batch_size, out_channels, classes_num)
        """

        x = F.relu_(self.bn1(self.conv1(input)))
        x = F.relu_(self.bn2(self.conv2(x)))

        if pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)

        return x


class AcousticModelCRnn8Dropout(nn.Module):
    def __init__(self, classes_num, midfeat, momentum):
        super(AcousticModelCRnn8Dropout, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=48, momentum=momentum)
        self.conv_block2 = ConvBlock(in_channels=48, out_channels=64, momentum=momentum)
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=96, momentum=momentum)
        self.conv_block4 = ConvBlock(in_channels=96, out_channels=128, momentum=momentum)

        self.fc5 = nn.Linear(midfeat, 384, bias=False)
        self.bn5 = nn.BatchNorm1d(384, momentum=momentum)
        self.prenet_emb = nn.Conv1d(128, 384, 1)

        self.gru = nn.GRU(input_size=768, hidden_size=256, num_layers=2,
                          bias=True, batch_first=True, dropout=0., bidirectional=True)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc5)
        init_bn(self.bn5)
        init_gru(self.gru)
        init_layer(self.fc)

    def forward(self, input, dyn_emb):
        """
        Args:
          input: (batch_size, channels_num, time_steps, freq_bins)

        Outputs:
          output: (batch_size, time_steps, classes_num)
        """

        x = self.conv_block1(input, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.transpose(1, 2).flatten(2)
        x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x = F.dropout(x, p=0.5, training=self.training, inplace=True)
        d = self.prenet_emb(dyn_emb.squeeze().permute(0, 2, 1)).permute(0, 2, 1)
        x = torch.cat([x, d], dim=-1)

        (x, _) = self.gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        output = torch.sigmoid(self.fc(x))
        return output


class Regress_onset_offset_frame_velocity_CRNN(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Regress_onset_offset_frame_velocity_CRNN, self).__init__()

        sample_rate = 16000
        window_size = 2048
        hop_size = 256
        mel_bins = 128
        fmin = 30
        fmax = sample_rate // 2

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        midfeat = 1024
        momentum = 0.01
        self.mulaw = MuLawEncoding(256)
        self.enc_dynamic = DynamicEncoder(embed_dim=128)

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size,
                                                 hop_length=hop_size, win_length=window_size, window=window,
                                                 center=center, pad_mode=pad_mode, freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate,
                                                 n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref,
                                                 amin=amin, top_db=top_db, freeze_parameters=True)

        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
        self.bn_dyn = nn.BatchNorm2d(mel_bins, momentum)

        self.frame_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum)
        self.reg_onset_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum)
        self.reg_offset_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum)
        self.velocity_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum)

        self.reg_onset_gru = nn.GRU(input_size=88 * 2, hidden_size=256, num_layers=1,
                                    bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.reg_onset_fc = nn.Linear(512, classes_num, bias=True)

        self.frame_gru = nn.GRU(input_size=88 * 3, hidden_size=256, num_layers=1,
                                bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.frame_fc = nn.Linear(512, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_gru(self.reg_onset_gru)
        init_gru(self.frame_gru)
        init_layer(self.reg_onset_fc)
        init_layer(self.frame_fc)

    def forward(self, input):
        """
        Args:
          input: (batch_size, data_length)

        Outputs:
          output_dict: dict, {
            'reg_onset_output': (batch_size, time_steps, classes_num),
            'reg_offset_output': (batch_size, time_steps, classes_num),
            'frame_output': (batch_size, time_steps, classes_num),
            'velocity_output': (batch_size, time_steps, classes_num)
          }
        """

        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)[:, :, :-1, :]  # (batch_size, 1, time_steps, mel_bins)
        with torch.no_grad():
            dyn_emb = self.enc_dynamic(self.mulaw(input.unsqueeze(1)).float()).unsqueeze(-1).transpose(1, -1)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = x / torch.max(x)

        dyn_emb = dyn_emb.transpose(1, 3)
        dyn_emb = self.bn_dyn(dyn_emb)
        dyn_emb = dyn_emb.transpose(1, 3)
        dyn_emb = dyn_emb / torch.max(dyn_emb)

        frame_output = self.frame_model(x, dyn_emb)  # (batch_size, time_steps, classes_num)
        reg_onset_output = self.reg_onset_model(x, dyn_emb)  # (batch_size, time_steps, classes_num)
        reg_offset_output = self.reg_offset_model(x, dyn_emb)  # (batch_size, time_steps, classes_num)
        velocity_output = self.velocity_model(x, dyn_emb)  # (batch_size, time_steps, classes_num)

        # Use velocities to condition onset regression
        x = torch.cat((reg_onset_output, (reg_onset_output ** 0.5) * velocity_output.detach()), dim=2)
        (x, _) = self.reg_onset_gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        reg_onset_output = torch.sigmoid(self.reg_onset_fc(x))
        """(batch_size, time_steps, classes_num)"""

        # Use onsets and offsets to condition frame-wise classification
        x = torch.cat((frame_output, reg_onset_output.detach(), reg_offset_output.detach()), dim=2)
        (x, _) = self.frame_gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        frame_output = torch.sigmoid(self.frame_fc(x))  # (batch_size, time_steps, classes_num)
        """(batch_size, time_steps, classes_num)"""

        output_dict = {
            'reg_onset_output': reg_onset_output,
            'reg_offset_output': reg_offset_output,
            'frame_output': frame_output,
            'velocity_output': velocity_output}

        return output_dict


class Regress_pedal_CRNN(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Regress_pedal_CRNN, self).__init__()

        sample_rate = 16000
        window_size = 2048
        hop_size = 256
        mel_bins = 80
        fmin = 30
        fmax = sample_rate // 2

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        midfeat = 640
        momentum = 0.01

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size,
                                                 hop_length=hop_size, win_length=window_size, window=window,
                                                 center=center, pad_mode=pad_mode, freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate,
                                                 n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref,
                                                 amin=amin, top_db=top_db, freeze_parameters=True)

        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)

        self.reg_pedal_onset_model = AcousticModelCRnn8Dropout(1, midfeat, momentum)
        self.reg_pedal_offset_model = AcousticModelCRnn8Dropout(1, midfeat, momentum)
        self.reg_pedal_frame_model = AcousticModelCRnn8Dropout(1, midfeat, momentum)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)

    def forward(self, input):
        """
        Args:
          input: (batch_size, data_length)

        Outputs:
          output_dict: dict, {
            'reg_onset_output': (batch_size, time_steps, classes_num),
            'reg_offset_output': (batch_size, time_steps, classes_num),
            'frame_output': (batch_size, time_steps, classes_num),
            'velocity_output': (batch_size, time_steps, classes_num)
          }
        """

        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        reg_pedal_onset_output = self.reg_pedal_onset_model(x)  # (batch_size, time_steps, classes_num)
        reg_pedal_offset_output = self.reg_pedal_offset_model(x)  # (batch_size, time_steps, classes_num)
        pedal_frame_output = self.reg_pedal_frame_model(x)  # (batch_size, time_steps, classes_num)

        output_dict = {
            'reg_pedal_onset_output': reg_pedal_onset_output,
            'reg_pedal_offset_output': reg_pedal_offset_output,
            'pedal_frame_output': pedal_frame_output}

        return output_dict


# This model is not trained, but is combined from the trained note and pedal models.
class Note_pedal(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        """The combination of note and pedal model.
        """
        super(Note_pedal, self).__init__()

        self.note_model = Regress_onset_offset_frame_velocity_CRNN(frames_per_second, classes_num)
        self.pedal_model = Regress_pedal_CRNN(frames_per_second, classes_num)

    def load_state_dict(self, m, strict=False):
        self.note_model.load_state_dict(m['note_model'], strict=strict)
        self.pedal_model.load_state_dict(m['pedal_model'], strict=strict)

    def forward(self, input):
        note_output_dict = self.note_model(input)
        pedal_output_dict = self.pedal_model(input)

        full_output_dict = {}
        full_output_dict.update(note_output_dict)
        full_output_dict.update(pedal_output_dict)
        return full_output_dict


class ResidualConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1):
        super(ResidualConv1d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(out_channel),
            nn.GELU(),
            nn.Conv1d(out_channel, out_channel, kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(out_channel)
        )

    def forward(self, x):
        return x + self.net(x)


class DynamicEncoder(nn.Module):
    def __init__(self, embed_dim=512, mode='dynamic'):
        super(DynamicEncoder, self).__init__()
        self.net = []
        self.net.append(nn.Sequential(
            nn.Conv1d(1 if mode == 'dynamic' else 512, 64, 3, padding=1),
            nn.Conv1d(64, 64, 9, stride=4, padding=4),
            nn.Dropout(p=0.5),
            nn.GELU()
        ))
        self.net.append(nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1),
            nn.Conv1d(128, 128, 9, stride=4, padding=4),
            nn.Dropout(p=0.5),
            nn.GELU()
        ))
        self.net.append(nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            ResidualConv1d(256, 256, 3, padding=1),
            nn.Conv1d(256, 256, 3, stride=2, padding=1),
            nn.Dropout(p=0.5),
            nn.GELU()
        ))
        self.net.append(nn.Sequential(
            nn.Conv1d(256, 256, 3, padding=1),
            ResidualConv1d(256, 256, 3, padding=1),
            nn.Conv1d(256, 256, 3, stride=2, padding=1),
            nn.Dropout(p=0.5),
            nn.GELU()
        ))
        self.net.append(nn.Sequential(
            nn.Conv1d(256, 512, 3, padding=1),
            ResidualConv1d(512, 512, 3, padding=1),
            nn.Conv1d(512, 512, 3, stride=2, padding=1),
            nn.Dropout(p=0.5),
            nn.GELU(),
        ))
        self.net.append(nn.Sequential(
            ResidualConv1d(512, 512, 3, padding=1),
            ResidualConv1d(512, 512, 3, padding=1),
            nn.Conv1d(512, 512, 3, stride=2, padding=1),
            nn.Dropout(p=0.5),
            nn.GELU(),
            nn.Conv1d(512, embed_dim, 1, 1, 0)
        ))
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(1)
        x = self.net(x)
        return x
