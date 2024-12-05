import torch
import torch.nn as nn
import math
from models import ResNet
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout=0, max_len=256):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class MyPositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens=128, max_len=256):
        super().__init__()
        self.pos = torch.linspace(0, 1, max_len).view(1, max_len, 1).repeat(1, 1, num_hiddens)
        for i in range(num_hiddens):
            i %= max_len
            self.pos[0, :, i] = torch.concatenate((self.pos[0, -i:, i], self.pos[0, :-i, i]))

        self.alpha = nn.Parameter(torch.randn((1, 1, num_hiddens)))

    def forward(self, X):
        # x_mean = X.mean(dim=1, keepdim=True)
        # pos = x_mean.expand(-1, X.shape[1], -1)*torch.linspace(0, 1, X.shape[1], device=X.device).view(1, X.shape[1], 1)
        pos = self.pos.to(device=X.device)*self.alpha
        X = X + pos
        return X


class CNNWise(nn.Module):
    """通道卷积"""
    def __init__(self, input_channel=2, output_channel=2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channel, output_channel*2, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(output_channel*2, output_channel, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(ffn_num, ffn_num*4)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num*4, ffn_num)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout=0, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = nn.functional.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, input_dim, output_dim, num_heads, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention()
        self.W_q = nn.Linear(input_dim, input_dim, bias=bias)
        self.W_k = nn.Linear(input_dim, input_dim, bias=bias)
        self.W_v = nn.Linear(input_dim, input_dim, bias=bias)
        self.W_o = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(x), self.num_heads)
        keys = transpose_qkv(self.W_k(x), self.num_heads)
        values = transpose_qkv(self.W_v(x), self.num_heads)


        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        # return self.W_o(output_concat)
        return output_concat


class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout=0, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        # (batch_size, channels, lens, features)
        return self.ln(self.dropout(Y) + X)


class BatchAddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, dropout=0, input_channel=1, output_channel=1, **kwargs):
        super(BatchAddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(output_channel)
        self.use_conv1 = input_channel != output_channel
        if self.use_conv1:
            self.conv1 = nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, X, Y):
        # (batch_size*features, channels, lens)
        if self.use_conv1:
            X = self.conv1(X)           # (batch_size*features, channels, lens)
        return self.bn(self.dropout(Y) + X)


class Block(nn.Module):
    """Transformer编码器块"""
    def __init__(self, input_dim, output_dim, num_heads, dropout=0):
        super().__init__()
        self.attention = MultiHeadAttention(input_dim, output_dim, num_heads)
        self.addnorm1 = AddNorm(output_dim, dropout)
        self.ffn = PositionWiseFFN(output_dim)
        self.addnorm2 = AddNorm(output_dim, dropout)

    def forward(self, X):
        Y = self.addnorm1(X, self.attention(X))
        return self.addnorm2(Y, self.ffn(Y))


class ChannelBlock(nn.Module):
    def __init__(self, input_dim, output_dim, input_channel, output_channel, num_heads, dropout=0):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.cnn = CNNWise(input_channel=input_channel, output_channel=output_channel)
        self.addbnorm = BatchAddNorm(input_channel=input_channel, output_channel=output_channel)
        self.blocks = nn.ModuleList()
        for i in range(self.output_channel):
            self.blocks.append(Block(input_dim, output_dim, num_heads))

    def forward(self, x):
        # (batch_size, channels, lens, features)
        b, c, l, f = x.shape
        x_re = x.permute(0, 3, 1, 2)  # (batch_size, features, channels, lens)
        x_re = x_re.reshape(b*f, c, l)  # (batch_size*features, channels, lens)

        x_channel = self.addbnorm(x_re, self.cnn(x_re))

        x_channel = x_channel.reshape(b, f, self.output_channel, l)    # (batch_size, features, channels, lens)
        x_channel = x_channel.permute(0, 2, 3, 1)       # (batch_size, channels, lens, features)

        out_blocks = []
        for i in range(self.output_channel):
            out_blocks.append(self.blocks[i](x_channel[:, i, :, :]))
        out_blocks = torch.stack(out_blocks, dim=1)               # (batch_size, channels, lens, features)

        return out_blocks


# 对步长进行二次卷积，形成多个通道, 逐位反馈层也有多个通道， 对七的优化
class Proposed(nn.Module):
    def __init__(self, outputs=53, channel=2, device='cuda:0'):
        super().__init__()
        # 2*1024
        self.RN = nn.Sequential(
            *ResNet.resnet_block1d3(4, 32, num_residuals=1),
            *ResNet.resnet_block1d3(32, 32, num_residuals=1),
            *ResNet.resnet_block1d3(32, 64, num_residuals=1),
            *ResNet.resnet_block1d3(64, 64, num_residuals=1),
            *ResNet.resnet_block1d3(64, 64, num_residuals=1)
        )

        self.pos = MyPositionalEncoding(num_hiddens=128, max_len=128)

        self.trans = nn.Sequential(
            ChannelBlock(input_dim=128, output_dim=128, input_channel=1, output_channel=channel, num_heads=2),
            *[ChannelBlock(input_dim=128, output_dim=128, input_channel=channel, output_channel=channel, num_heads=2) for _ in range(5)]
        )



        self.labelnet = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*channel, outputs)
        )

        self.window = torch.hann_window(32, device=device)  # 使用 Hann 窗函数

    def forward(self, x):  # (batch_size, lens, feature)

        # 计算幅度与相位
        X_commplex = x[:, :, 0] + x[:, :, 1] * 1j
        x_AP = torch.stack((torch.abs(X_commplex), torch.angle(X_commplex)), dim=2)

        # 计算短时傅里叶变换 (STFT)
        stft_result = torch.stft(X_commplex, n_fft=32, hop_length=32, win_length=32,
                                 window=self.window,
                                 return_complex=True)[:, :, :128].permute(0, 2, 1)
        x_stft = torch.concatenate((torch.abs(stft_result), torch.angle(stft_result)), dim=2)

        x = torch.concatenate([x, x_AP], dim=2)
        x = x.permute(0, 2, 1)
        x = self.RN(x)  # (batch_size, channel, lens)

        x = x.permute(0, 2, 1)  # (batch_size, lens, features)

        x = torch.concatenate((x, x_stft), dim=2)

        x = self.pos(x)         # (batch_size, lens, features)
        x = x.unsqueeze(1)      # (batch_size, channel, lens, features)
        out_trans = self.trans(x)  # (batch_size, channel, lens, features)

        outputs = self.labelnet(out_trans[:, :, 0, :])  # (batch_size, outputs)
        return outputs


if __name__ == '__main__':
    device = 'cpu'
    encoder = Proposed(device=device).to(device=device)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    tem = torch.ones(3, 4096, 2, dtype=torch.float32, device=device)
    outputs = encoder(tem)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    outputs = torch.mean(outputs, dim=1)
    outputs = torch.sum(outputs)
    outputs.backward()
    torch.save(encoder.state_dict(), './tem.pth')
    pass
