import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
from ncps.torch import CfC

RNN_TYPE = "GRU"
bCrossOver = True
class ERB(nn.Module):
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft // 2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs - erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs - erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz):
        erb_f = 21.4 * np.log10(0.00437 * freq_hz + 1)
        return erb_f

    def erb2hz(self, erb_f):
        freq_hz = (10 ** (erb_f / 21.4) - 1) / 0.00437
        return freq_hz

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        low_lim = erb_subband_1 / nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points) / fs * nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        erb_filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) \
                                          / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2 - 2):
            erb_filters[i + 1, bins[i]:bins[i + 1]] = (np.arange(bins[i], bins[i + 1]) - bins[i] + 1e-12) \
                                                      / (bins[i + 1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i + 1]:bins[i + 2]] = (bins[i + 2] - np.arange(bins[i + 1], bins[i + 2]) + 1e-12) \
                                                          / (bins[i + 2] - bins[i + 1] + 1e-12)

        erb_filters[-1, bins[-2]:bins[-1] + 1] = 1 - erb_filters[-2, bins[-2]:bins[-1] + 1]

        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))

    def bm(self, x):
        """x: (B,C,T,F)"""
        x_low = x[..., :self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1:])
        return torch.cat([x_low, x_high], dim=-1)

    def bs(self, x_erb):
        """x: (B,C,T,F_erb)"""
        x_erb_low = x_erb[..., :self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


class SFE(nn.Module):
    """Subband Feature Extraction"""

    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, (kernel_size - 1) // 2))

    def forward(self, x):
        """x: (B,C,T,F)"""
        xs = self.unfold(x).reshape(x.shape[0], x.shape[1] * self.kernel_size, x.shape[2], x.shape[3])
        return xs


class TRA(nn.Module):
    """Temporal Recurrent Attention"""

    def __init__(self, channels):
        super().__init__()
        if RNN_TYPE == "CFC":
            self.att_rnn = CfC(channels, channels * 2, mode="pure", batch_first=True)
        elif RNN_TYPE == "LSTM":
            self.att_rnn = nn.LSTM(channels, channels * 2, batch_first=True)
        else:
            self.att_rnn = nn.GRU(channels, channels * 2, 1, batch_first=True)

        self.att_fc = nn.Linear(channels * 2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        """x: (B,C,T,F)"""
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        at = self.att_rnn(zt.transpose(1, 2))[0]
        at = self.att_fc(at).transpose(1, 2)
        at = self.att_act(at)
        At = at[..., None]  # (B,C,T,1)

        return x * At


class TFCM_cell(nn.Module):
    """Group Temporal Convolution"""

    def __init__(self, channels, kernel_size, bDecConv=False):
        super().__init__()
        if bDecConv == True:
            self.point_conv1 = nn.ConvTranspose2d(channels, channels, (1, 1))
            self.point_bn1 = nn.BatchNorm2d(channels)
            self.point_act1 = nn.PReLU()

            self.dpth_pad1 = nn.ConstantPad2d((0, 0, 2 * (kernel_size[1] - 1), 0), 0.0)
            self.depth_conv1 = nn.ConvTranspose2d(channels, channels, kernel_size,
                                                  1, padding=(2 * 2, 1), dilation=(2, 1), groups=channels)

            self.depth_bn1 = nn.BatchNorm2d(channels)
            self.depth_act1 = nn.PReLU()

            self.point_conv2 = nn.ConvTranspose2d(channels, channels, (1, 1))
            self.point_bn2 = nn.BatchNorm2d(channels)
            self.point_act2 = nn.PReLU()

            self.dpth_pad2 = nn.ConstantPad2d((0, 0, (kernel_size[1] - 1), 0), 0.0)
            self.depth_conv2 = nn.ConvTranspose2d(channels, channels, kernel_size,
                                                  1, padding=(2 * 1, 1), dilation=(1, 1), groups=channels)

            self.depth_bn2 = nn.BatchNorm2d(channels)
            self.depth_act2 = nn.PReLU()

            self.point_conv3 = nn.ConvTranspose2d(channels, channels, (1, 1))
            self.depth_bn3 = nn.BatchNorm2d(channels)
        else:
            self.point_conv1 = nn.Conv2d(channels, channels, (1, 1))
            self.point_bn1 = nn.BatchNorm2d(channels)
            self.point_act1 = nn.PReLU()

            self.dpth_pad1 = nn.ConstantPad2d((0, 0, (kernel_size[1] - 1), 0), 0.0)
            self.depth_conv1 = nn.Conv2d(channels, channels, kernel_size,
                                         1, padding=(0, 1), dilation=(1, 1), groups=channels)
            self.depth_bn1 = nn.BatchNorm2d(channels)
            self.depth_act1 = nn.PReLU()

            self.point_conv2 = nn.Conv2d(channels, channels, (1, 1))
            self.point_bn2 = nn.BatchNorm2d(channels)
            self.point_act2 = nn.PReLU()

            self.dpth_pad2 = nn.ConstantPad2d((0, 0, 2 * (kernel_size[1] - 1), 0), 0.0)
            self.depth_conv2 = nn.Conv2d(channels, channels, kernel_size,
                                         1, padding=(0, 1), dilation=(2, 1), groups=channels)
            self.depth_bn2 = nn.BatchNorm2d(channels)
            self.depth_act2 = nn.PReLU()

            self.point_conv3 = nn.Conv2d(channels, channels, (1, 1))
            self.depth_bn3 = nn.BatchNorm2d(channels)

    def forward(self, x):
        """x: (B, C, T, F)"""
        h1 = self.point_act1(self.point_bn1(self.point_conv1(x)))
        h1 = self.dpth_pad1(h1)
        h1 = self.depth_act1(self.depth_bn1(self.depth_conv1(h1)))

        h1 = self.point_act2(self.point_bn2(self.point_conv2(h1)))
        h1 = self.dpth_pad2(h1)
        h1 = self.depth_act2(self.depth_bn2(self.depth_conv2(h1)))

        h1 = self.depth_bn3(self.point_conv3(h1))
        x = x + h1
        return x


class TFCM(nn.Module):
    """Group Temporal Convolution"""

    def __init__(self, channels, kernel_size, bDecConv=False):
        super().__init__()
        self.TFCM_cell = TFCM_cell(channels, kernel_size, bDecConv)

        if RNN_TYPE == "CFC":
            self.att_rnn = CfC(channels, channels * 2, mode="pure", batch_first=True)
        elif RNN_TYPE == "LSTM":
            self.att_rnn = nn.LSTM(channels, channels * 2, 1, batch_first=True)
        else:
            self.att_rnn = nn.GRU(channels, channels * 2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels * 2, channels)
        self.att_act = nn.Sigmoid()

    def shuffle(self, x1, x2):
        """x1, x2: (B,C,T,F)"""
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()  # (B,C,2,T,F)
        x = rearrange(x, 'b c g t f -> b (c g) t f')  # (B,2C,T,F)
        return x

    def forward(self, x):
        """x: (B, C, T, F)"""
        h1 = self.TFCM_cell(x)
        zt = torch.mean(h1.pow(2), dim=-1)  # (B,C,T)
        at = self.att_rnn(zt.transpose(1, 2))[0]
        at = self.att_act(self.att_fc(at))
        at = at.transpose(1, 2)
        At = at[..., None]  # (B,C,T,1)
        h2 = h1 * At
        x1, x2 = torch.chunk(h2, chunks=2, dim=1)
        x = self.shuffle(x1, x2)
        return x


class LeCun(nn.Module):
    def __init__(self):
        super(LeCun, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_deconv=False,
                 is_last=False):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = LeCun() if is_last else nn.PReLU()

    def forward(self, x):
        ret = self.act(self.bn(self.conv(x)))
        return ret


class CGTFCM(nn.Module):
    """Group Temporal Convolution"""

    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False,
                 bLastLayer=False):
        super().__init__()
        self.use_deconv = use_deconv
        self.bLastLayer = bLastLayer
        self.pad_size = (kernel_size[0] - 1) * dilation[0]
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.sfe = SFE(kernel_size=3, stride=1)
        self.point_conv1 = conv_module(in_channels // 4 * 3, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)

        self.depth_conv = conv_module(hidden_channels, hidden_channels, kernel_size,
                                      stride=stride, padding=padding,
                                      dilation=dilation, groups=hidden_channels // 2)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)

        self.point_conv2 = conv_module(hidden_channels, in_channels // 4, 1)
        self.point_bn2 = nn.BatchNorm2d(in_channels // 4)
        self.point_act = nn.PReLU()
        if not bLastLayer:
            self.act2 = nn.PReLU()

        self.tra = TRA(in_channels // 4)

    def shuffle(self, x):
        """x1, x2: (B,C,T,F)"""
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()  # (B,C,2,T,F)
        x = rearrange(x, 'b c g t f -> b (c g) t f')  # (B,2C,T,F)
        return x

    def forward(self, x):
        """x: (B, C, T, F)"""
        x1, x2, x3, x4 = torch.chunk(x, chunks=4, dim=1)

        sfe = self.sfe(x4)
        h1 = self.point_bn1(self.point_conv1(sfe))
        h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])
        h1 = self.depth_bn(self.depth_conv(h1))
        h2, h3 = torch.chunk(h1, chunks=2, dim=1)
        h4 = torch.cat([x3, h2], dim=1)
        if not self.bLastLayer:
            x3 = self.act2(h3)
        h4 = self.point_act(self.point_bn2(self.point_conv2(h4)))
        h4 = x4 + h4
        h4 = self.tra(h4)
        x5 = torch.cat([h4, x1, x2, x3], dim=1)
        # x = self.shuffle(x5)
        return x5


class GRNN(nn.Module):
    """Grouped RNN"""

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.rnn1 = nn.GRU(input_size // 2, hidden_size // 2, num_layers, batch_first=batch_first,
                           bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size // 2, hidden_size // 2, num_layers, batch_first=batch_first,
                           bidirectional=bidirectional)

    def forward(self, x, h=None):
        """
        x: (B, seq_length, input_size)
        h: (num_layers, B, hidden_size)
        """
        if h == None:
            if self.bidirectional:
                h = torch.zeros(self.num_layers * 2, x.shape[0], self.hidden_size, device=x.device)
            else:
                h = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=x.device)
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        h1, h2 = h1.contiguous(), h2.contiguous()
        y1, h1 = self.rnn1(x1, h1)
        y2, h2 = self.rnn2(x2, h2)
        y = torch.cat([y1, y2], dim=-1)
        h = torch.cat([h1, h2], dim=-1)
        return y, h


class DPGRNN(nn.Module):
    """Grouped Dual-path RNN"""

    def __init__(self, input_size, width, hidden_size, **kwargs):
        super(DPGRNN, self).__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size

        self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size // 2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm(((width, hidden_size)), eps=1e-8)

    def forward(self, x):
        """x: (B, C, T, F)"""
        ## Intra RNN
        x = x.permute(0, 2, 3, 1)  # (B,T,F,C)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (B*T,F,C)
        intra_x = self.intra_rnn(intra_x)[0]  # (B*T,F,C)
        intra_x = self.intra_fc(intra_x)  # (B*T,F,C)
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.hidden_size)  # (B,T,F,C)
        intra_x = self.intra_ln(intra_x)
        intra_out = torch.add(x, intra_x)

        ## Inter RNN
        x = intra_out.permute(0, 2, 1, 3)  # (B,F,T,C)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        inter_x = self.inter_rnn(inter_x)[0]  # (B*F,T,C)
        inter_x = self.inter_fc(inter_x)  # (B*F,T,C)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.hidden_size)  # (B,F,T,C)
        inter_x = inter_x.permute(0, 2, 1, 3)  # (B,T,F,C)
        inter_x = self.inter_ln(inter_x)
        inter_out = torch.add(intra_out, inter_x)

        dual_out = inter_out.permute(0, 3, 1, 2)  # (B,C,T,F)

        return dual_out


class GRNN_CFC(nn.Module):
    """Grouped RNN"""

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        if bidirectional:
            self.rnn1 = CfC(input_size, hidden_size, mode="pure",
                            batch_first=True)  # nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)
            self.rnn2 = CfC(input_size, hidden_size, mode="pure",
                            batch_first=True)  # nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)
        else:
            self.rnn1 = CfC(input_size, hidden_size, mode="pure", batch_first=True)

    def forward(self, x, h=None):

        if self.bidirectional:
            xn = torch.flip(x, dims=[1])
            y1 = self.rnn1(x)[0]
            y1n = self.rnn2(xn)[0]
            y = torch.cat([y1, y1n], dim=-1)
        else:
            y = self.rnn1(x)[0]
        return y


class DPGRNN_CFC(nn.Module):
    """Grouped Dual-path RNN"""

    def __init__(self, input_size, width, hidden_size, **kwargs):
        super(DPGRNN_CFC, self).__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size
        """
        帧内频率循环处理，每一次从低到高，然后从高到低完成一次后中间状态清零，不同时间分成多个batch，独立处理
        """
        self.intra_rnn = GRNN_CFC(input_size=input_size, hidden_size=hidden_size // 2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)
        """
        帧间单次处理，由于时间在第二维度，一次调用全部处理完成
        """
        self.inter_rnn = GRNN_CFC(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm(((width, hidden_size)), eps=1e-8)

    def forward(self, x):
        """x: (B, C, T, F)"""
        ## Intra RNN
        x = x.permute(0, 2, 3, 1)  # (B,T,F,C)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (B*T,F,C)
        intra_x = self.intra_rnn(intra_x)  # (B*T,F,C)
        intra_x = self.intra_fc(intra_x)  # (B*T,F,C)
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.hidden_size)  # (B,T,F,C)
        intra_x = self.intra_ln(intra_x)
        intra_out = torch.add(x, intra_x)

        ## Inter RNN
        x = intra_out.permute(0, 2, 1, 3)  # (B,F,T,C)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        inter_x = self.inter_rnn(inter_x)  # (B*F,T,C)
        inter_x = self.inter_fc(inter_x)  # (B*F,T,C)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.hidden_size)  # (B,F,T,C)
        inter_x = inter_x.permute(0, 2, 1, 3)  # (B,T,F,C)
        inter_x = self.inter_ln(inter_x)
        inter_out = torch.add(intra_out, inter_x)

        dual_out = inter_out.permute(0, 3, 1, 2)  # (B,C,T,F)

        return dual_out


class GRNN_LSTM(nn.Module):
    """Grouped RNN"""

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        if bidirectional:
            self.rnn1 = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=batch_first)  # nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)
            self.rnn2 = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=batch_first)  # nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)
        else:
            self.rnn1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)

    def forward(self, x, h=None):

        if self.bidirectional:
            xn = torch.flip(x, dims=[1])
            y1 = self.rnn1(x)[0]
            y1n = self.rnn2(xn)[0]
            y = torch.cat([y1, y1n], dim=-1)
        else:
            y = self.rnn1(x)[0]
        return y


class DPGRNN_LSTM(nn.Module):
    """Grouped Dual-path RNN"""

    def __init__(self, input_size, width, hidden_size, **kwargs):
        super(DPGRNN_LSTM, self).__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size
        """
        帧内频率循环处理，每一次从低到高，然后从高到低完成一次后中间状态清零，不同时间分成多个batch，独立处理
        """
        self.intra_rnn = GRNN_LSTM(input_size=input_size, hidden_size=hidden_size // 2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)
        """
        帧间单次处理，由于时间在第二维度，一次调用全部处理完成
        """
        self.inter_rnn = GRNN_LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm(((width, hidden_size)), eps=1e-8)

    def forward(self, x):
        """x: (B, C, T, F)"""
        ## Intra RNN
        x = x.permute(0, 2, 3, 1)  # (B,T,F,C)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (B*T,F,C)
        intra_x = self.intra_rnn(intra_x)  # (B*T,F,C)
        intra_x = self.intra_fc(intra_x)  # (B*T,F,C)
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.hidden_size)  # (B,T,F,C)
        intra_x = self.intra_ln(intra_x)
        intra_out = torch.add(x, intra_x)

        ## Inter RNN
        x = intra_out.permute(0, 2, 1, 3)  # (B,F,T,C)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        inter_x = self.inter_rnn(inter_x)  # (B*F,T,C)
        inter_x = self.inter_fc(inter_x)  # (B*F,T,C)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.hidden_size)  # (B,F,T,C)
        inter_x = inter_x.permute(0, 2, 1, 3)  # (B,T,F,C)
        inter_x = self.inter_ln(inter_x)
        inter_out = torch.add(intra_out, inter_x)

        dual_out = inter_out.permute(0, 3, 1, 2)  # (B,C,T,F)

        return dual_out


class Encoder(nn.Module):
    def __init__(self, layerCnt = 4):
        super().__init__()
        if layerCnt == 8:
            self.en_convs = nn.ModuleList([
                ConvBlock(3, 16, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=False, is_last=False),
                ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=False, is_last=False),
                # TFCM(16, (3, 3), bDecConv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(3, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(4, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(6, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(7, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(8, 1), use_deconv=False, bLastLayer=True)
            ])
        elif layerCnt == 7:
            self.en_convs = nn.ModuleList([
                ConvBlock(3, 16, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=False, is_last=False),
                ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=False, is_last=False),
                # TFCM(16, (3, 3), bDecConv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(3, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(4, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(6, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(7, 1), use_deconv=False, bLastLayer=True)
            ])
        elif layerCnt == 6:
            self.en_convs = nn.ModuleList([
                ConvBlock(3, 16, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=False, is_last=False),
                ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=False, is_last=False),
                # TFCM(16, (3, 3), bDecConv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(3, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(4, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(6, 1), use_deconv=False, bLastLayer=True)
            ])
        elif layerCnt == 5:
            self.en_convs = nn.ModuleList([
                ConvBlock(3, 16, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=False, is_last=False),
                ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=False, is_last=False),
                # TFCM(16, (3, 3), bDecConv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(3, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(4, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1), use_deconv=False, bLastLayer=True)
            ])
        elif layerCnt == 4:
            self.en_convs = nn.ModuleList([
                ConvBlock(3, 16, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=False, is_last=False),
                ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=False, is_last=False),
                # TFCM(16, (3, 3), bDecConv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(3, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1), use_deconv=False, bLastLayer=True)
            ])
        elif layerCnt == 3:
            self.en_convs = nn.ModuleList([
                ConvBlock(3, 16, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=False, is_last=False),
                ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=False, is_last=False),
                # TFCM(16, (3, 3), bDecConv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(4, 1), use_deconv=False, bLastLayer=True)
            ])
        elif layerCnt == 2:
            self.en_convs = nn.ModuleList([
                ConvBlock(3, 16, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=False, is_last=False),
                ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=False, is_last=False),
                # TFCM(16, (3, 3), bDecConv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), use_deconv=False),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(3, 1), use_deconv=False, bLastLayer=True)
            ])

    def forward(self, x):
        en_outs = []
        for i in range(len(self.en_convs)):
            x = self.en_convs[i](x)
            en_outs.append(x)
        return x, en_outs


class Decoder(nn.Module):
    def __init__(self, layerCnt = 4):
        super().__init__()
        if layerCnt == 8:
            self.de_convs = nn.ModuleList([
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 8, 1), dilation=(8, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 7, 1), dilation=(7, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 6, 1), dilation=(6, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 5, 1), dilation=(5, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 4, 1), dilation=(4, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 3, 1), dilation=(3, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 2, 1), dilation=(2, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 1, 1), dilation=(1, 1), use_deconv=True,
                       bLastLayer=True),
                # TFCM(16, (3, 3), bDecConv=True),
                ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=True, is_last=False),
                ConvBlock(16, 2, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=True, is_last=True)
            ])
        if layerCnt == 7:
            self.de_convs = nn.ModuleList([
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 7, 1), dilation=(7, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 6, 1), dilation=(6, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 5, 1), dilation=(5, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 4, 1), dilation=(4, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 3, 1), dilation=(3, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 2, 1), dilation=(2, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 1, 1), dilation=(1, 1), use_deconv=True,
                       bLastLayer=True),
                # TFCM(16, (3, 3), bDecConv=True),
                ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=True, is_last=False),
                ConvBlock(16, 2, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=True, is_last=True)
            ])
        elif layerCnt == 6:
            self.de_convs = nn.ModuleList([
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 6, 1), dilation=(6, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 5, 1), dilation=(5, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 4, 1), dilation=(4, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 3, 1), dilation=(3, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 2, 1), dilation=(2, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 1, 1), dilation=(1, 1), use_deconv=True, bLastLayer=True),
                # TFCM(16, (3, 3), bDecConv=True),
                ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=True, is_last=False),
                ConvBlock(16, 2, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=True, is_last=True)
            ])
        elif layerCnt == 5:
            self.de_convs = nn.ModuleList([
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 5, 1), dilation=(5, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 4, 1), dilation=(4, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 3, 1), dilation=(3, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 2, 1), dilation=(2, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 1, 1), dilation=(1, 1), use_deconv=True, bLastLayer=True),
                # TFCM(16, (3, 3), bDecConv=True),
                ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=True, is_last=False),
                ConvBlock(16, 2, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=True, is_last=True)
            ])
        elif layerCnt == 4:
            self.de_convs = nn.ModuleList([
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 5, 1), dilation=(5, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 3, 1), dilation=(3, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 2, 1), dilation=(2, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 1, 1), dilation=(1, 1), use_deconv=True,
                       bLastLayer=True),
                # TFCM(16, (3, 3), bDecConv=True),
                ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=True, is_last=False),
                ConvBlock(16, 2, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=True, is_last=True)
            ])
        elif layerCnt == 3:
            self.de_convs = nn.ModuleList([
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 4, 1), dilation=(4, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 2, 1), dilation=(2, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 1, 1), dilation=(1, 1), use_deconv=True,
                       bLastLayer=True),
                # TFCM(16, (3, 3), bDecConv=True),
                ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=True, is_last=False),
                ConvBlock(16, 2, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=True, is_last=True)
            ])
        elif layerCnt == 2:
            self.de_convs = nn.ModuleList([
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 3, 1), dilation=(3, 1), use_deconv=True),
                CGTFCM(16, 8, (3, 3), stride=(1, 1), padding=(2 * 1, 1), dilation=(1, 1), use_deconv=True,
                       bLastLayer=True),
                # TFCM(16, (3, 3), bDecConv=True),
                ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=True, is_last=False),
                ConvBlock(16, 2, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=True, is_last=True)
            ])

    def forward(self, x, en_outs):
        N_layers = len(self.de_convs)
        for i in range(N_layers):
            x = self.de_convs[i](x + en_outs[N_layers - 1 - i])
        return x


class Mask(nn.Module):
    """Complex Ratio Mask"""

    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        s_real = spec[:, 0] * mask[:, 0] - spec[:, 1] * mask[:, 1]
        s_imag = spec[:, 1] * mask[:, 0] + spec[:, 0] * mask[:, 1]
        s = torch.stack([s_real, s_imag], dim=1)  # (B,2,T,F)
        return s


class CGTFCRN(nn.Module):
    def __init__(self, convCnt = 4):
        super().__init__()
        self.erb = ERB(65, 64)
        self.encoder = Encoder(convCnt)
        if RNN_TYPE == "CFC":
            self.dpgrnn1 = DPGRNN_CFC(16, 33, 16)
            self.dpgrnn2 = DPGRNN_CFC(16, 33, 16)
        elif RNN_TYPE == "LSTM":
            self.dpgrnn1 = DPGRNN_LSTM(16, 33, 16)
            self.dpgrnn2 = DPGRNN_LSTM(16, 33, 16)
        else:
            self.dpgrnn1 = DPGRNN(16, 33, 16)
            self.dpgrnn2 = DPGRNN(16, 33, 16)
        self.decoder = Decoder(convCnt)
        self.mask = Mask()

    def forward(self, spec):
        """
        spec: (B, F, T, 2)
        """
        spec_ref = spec  # (B,F,T,2)

        spec_real = spec[..., 0].permute(0, 2, 1)
        spec_imag = spec[..., 1].permute(0, 2, 1)
        spec_mag = torch.sqrt(spec_real ** 2 + spec_imag ** 2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,257)

        feat = self.erb.bm(feat)  # (B,3,T,129)

        feat, en_outs = self.encoder(feat)

        feat = self.dpgrnn1(feat)  # (B,16,T,33)
        feat = self.dpgrnn2(feat)  # (B,16,T,33)

        m_feat = self.decoder(feat, en_outs)

        m = self.erb.bs(m_feat)

        spec_enh = self.mask(m, spec_ref.permute(0, 3, 2, 1))  # (B,2,T,F)
        spec_enh = spec_enh.permute(0, 3, 2, 1)  # (B,F,T,2)

        return spec_enh


if __name__ == "__main__":
    model = CGTFCRN().eval()

    """complexity count"""
    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(model, (257, 63, 2), as_strings=True,
                                              print_per_layer_stat=True, verbose=True)
    print(flops, params)

    """causality check"""
    a = torch.randn(1, 160000)
    b = torch.randn(1, 160000)
    c = torch.randn(1, 160000)
    x1 = torch.cat([a, b], dim=1)
    x2 = torch.cat([a, c], dim=1)

    x1 = torch.stft(x1, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    x2 = torch.stft(x2, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    y1 = model(x1)[0]
    y2 = model(x2)[0]
    y1 = torch.istft(y1, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    y2 = torch.istft(y2, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)

    print((y1[:16000 - 256 * 2] - y2[:16000 - 256 * 2]).abs().max())
    print((y1[16000:] - y2[16000:]).abs().max())
