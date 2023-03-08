import numpy as np
import torch
import torch.nn as nn

from .util import single_as_batch


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
        self._conv1 = SlimConv2d(i_channel, o_channel,
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
                res = ResidualBlock(i_channel=out_ch, o_channel=out_ch,
                                    in_size=out_size)
                conv_layers.append(res)

            padding, out_size = res.padding, res.out_size
            in_ch = out_ch

        conv_layers.append(nn.ReLU(inplace=True))
        self.resnet = nn.Sequential(*conv_layers)

    def forward(self, x):
        out = self.resnet(x)
        return out


class MLPEncoder(nn.Module):
    def __init__(self, n_input, n_hiddens):
        super().__init__()

        assert len(n_hiddens) > 0

        self._linear = nn.ModuleList()
        n_prev = n_input
        for n_curr in n_hiddens:
            self._linear.append(nn.Linear(n_prev, n_curr))
            self._linear.append(nn.ReLU(inplace=True))
            n_prev = n_curr

    def forward(self, x: torch.Tensor):
        for layer in self._linear:
            x = layer(x)
        return x


class TileEncoder(nn.Module):
    def __init__(self, in_ch, in_size):
        super().__init__()

        self.tile_net = ResNet(
            in_ch, in_size, channel_and_blocks=[[32, 2], [32, 2], [64, 2]])

        sample_in = torch.zeros(1, in_ch, *in_size)
        with torch.no_grad():
            self.h_size = len(self.tile_net(sample_in).flatten())

    def forward(self, x):
        bs, na = x['tile'].shape[:2]
        x_tile = x['tile'].view(-1, *x['tile'].shape[2:])
        h_tile = self.tile_net(x_tile)
        h_tile = h_tile.view(bs, na, -1)  # flatten
        return h_tile


class ItemEncoder(nn.Module):
    def __init__(self, n_item_hidden):
        super().__init__()

        n_item_type = 17 + 1  # the additional one stands for no item
        n_item_feature = 11
        n_type_hidden = 32
        n_fc_in = n_item_feature + n_type_hidden
        self.type_embedding = nn.Embedding(n_item_type, n_type_hidden)
        self.post_fc = MLPEncoder(n_fc_in, n_hiddens=[n_item_hidden])

    def forward(self, x):
        bs, na, ni = x['item_type'].shape  # batch size, num agent, num item
        x_item_type = x['item_type'].long().view(-1, ni)
        h_type_embed = self.type_embedding(x_item_type).view(bs, na, ni, -1)
        h_item = torch.cat([h_type_embed, x['item']], dim=-1)
        h_item = self.post_fc(h_item)
        h_item = h_item.max(dim=2)[0]
        return h_item


class PrevActionEncoder(nn.Module):
    def __init__(self, n_legal, n_embed):
        super().__init__()

        self.n_legal = n_legal
        self.embedding = nn.ModuleDict({
            name: nn.Embedding(n_input, n_embed)
            for name, n_input in n_legal.items()
        })

    def forward(self, x):
        bs, na, nh = x['prev_act'].shape  # batch size, num actions, num action heads
        x_prev_act = x['prev_act'].long().view(-1, nh)
        h_embed = [
            self.embedding[name](x_prev_act[:, i])
            for i, name in enumerate(self.n_legal)
        ]
        h_embed = torch.cat(h_embed, dim=-1).view(bs, na, -1)
        return h_embed


class SelfEncoder(nn.Module):
    def __init__(self, in_img_ch, in_img_size, n_self_feat,
                 n_legal, n_self_hidden):
        super().__init__()

        prev_act_embed_size = 16
        prev_act_hidden_size = len(n_legal) * prev_act_embed_size
        item_hidden_size = 128
        self.prev_act_embed = PrevActionEncoder(n_legal, prev_act_embed_size)
        self.item_net = ItemEncoder(item_hidden_size)
        self.img_net = TileEncoder(in_img_ch, in_img_size)
        mlp_input_size = n_self_feat + self.img_net.h_size + \
            prev_act_hidden_size + item_hidden_size
        self.mlp_net = MLPEncoder(mlp_input_size, n_hiddens=[n_self_hidden])

    def forward(self, x):
        bs, na, _ = x['team'].shape
        h_tile = self.img_net(x)
        h_pre_act = self.prev_act_embed(x)
        h_item = self.item_net(x)
        h_self = torch.cat([
            h_tile,
            x['team'],
            h_item,
            h_pre_act,
            *x['legal'].values(),
            x['game'].unsqueeze(1).expand(-1, na, -1),
        ], dim=-1)
        h_self = self.mlp_net(h_self)
        return h_self


class EntityEncoder(nn.Module):
    def __init__(self, entity_type, n_entity_feat, n_attn_hidden):
        super().__init__()

        self.entity_type = entity_type
        self.n_self_hidden_used = n_attn_hidden
        self.net = MLPEncoder(n_attn_hidden + n_entity_feat,
                              n_hiddens=[n_attn_hidden])

    def forward(self, x, h_self):
        x_entity = x[self.entity_type]
        bs, na, ne, nf = x_entity.shape

        h_self = h_self.unsqueeze(2).expand(-1, -1, ne, -1)
        h_self = h_self[:, :, :, :self.n_self_hidden_used]
        h_entity = torch.cat([h_self, x_entity], dim=-1)
        h_entity = self.net(h_entity)
        return h_entity


class InteractionBlock(nn.Module):
    def __init__(self, n_attn_hidden):
        super().__init__()

        self.n_attn_hidden = n_attn_hidden
        encoder_block = nn.TransformerEncoderLayer(
            d_model=n_attn_hidden, nhead=2, dim_feedforward=512, dropout=0.)
        self.transformer = nn.TransformerEncoder(encoder_block, num_layers=3)

    def forward(self, x, h_self, *hs_other):
        h_self = h_self.unsqueeze(2)[:, :, :, :self.n_attn_hidden]
        h = torch.cat([h_self, *hs_other], dim=2)
        bs, na, ne, nf = h.shape  # batch_size, num_agent, num_entity, num_feature
        mask = torch.cat([x['team_mask'], x['npc_mask'], x['enemy_mask']], dim=-1)
        mask = mask.to(bool).view(bs * na, ne)
        h = h.view(bs * na, ne, nf).transpose(0, 1)
        h = self.transformer(h, src_key_padding_mask=mask)
        h = h.transpose(0, 1).view(bs, na, ne, nf)
        h = h.max(dim=2)[0]
        return h


class MemoryBlock(nn.Module):
    def __init__(self, n_attn_hidden, n_lstm_hidden):
        super().__init__()

        self.n_attn_hidden = n_attn_hidden
        self.lstm = nn.LSTMCell(n_lstm_hidden, n_lstm_hidden)

    def forward(self, x, h_self, h_inter, hxs, cxs, seq_len):
        h = torch.cat([h_self[:, :, self.n_attn_hidden:], h_inter], dim=-1)
        bs, na, nf = h.shape  # batch_size, num_agent, num_feature
        nt = bs // seq_len  # num of truncation
        resets = x['reset'].repeat(1, na)
        h = h.view(nt, seq_len, na, -1).transpose(1, 0).reshape(seq_len, nt * na, -1)
        hys = [hxs[::seq_len].reshape(-1, nf)]  # [nt * na, nf]
        cys = [cxs[::seq_len].reshape(-1, nf)]
        for i in range(seq_len):
            hys[i] = hys[i] * (1 - resets[i::seq_len]).reshape(nt * na, 1)
            cys[i] = cys[i] * (1 - resets[i::seq_len]).reshape(nt * na, 1)
            hy, cy = self.lstm(h[i], (hys[i], cys[i]))
            hys.append(hy)
            cys.append(cy)
        hys = torch.stack(hys[1:], dim=0)  # [seq_len, nt * na, nf]
        cys = torch.stack(cys[1:], dim=0)
        hys = hys.view(seq_len, nt, na, -1).transpose(1, 0).reshape(bs, na, -1)
        cys = cys.view(seq_len, nt, na, -1).transpose(1, 0).reshape(bs, na, -1)
        return hys, cys


class PolicyHead(nn.Module):
    def __init__(self, n_hidden, n_legal):
        super().__init__()

        self.heads = nn.ModuleDict({
            name: nn.Linear(n_hidden, n_output)
            for name, n_output in n_legal.items()
        })

    def forward(self, h):
        logits = {name: self.heads[name](h) for name in self.heads}
        return logits


class NMMONet(nn.Module):
    def __init__(self):
        super().__init__()

        in_ch = 7
        in_size = [25, 25]
        n_player_feat = 262
        n_game_feat = 26
        n_legal = {
            'move': 5,
            'target': 19,
            'use': 3,
            'sell': 3,
        }
        n_self_feat = n_player_feat + n_game_feat + sum(n_legal.values())
        n_npc_feat = n_enemy_feat = 30

        self.n_attn_hidden = n_attn_hidden = n_ally_hidden = 256
        self.n_lstm_hidden = n_lstm_hidden = n_self_hidden = 512

        self.self_net = SelfEncoder(in_ch, in_size, n_self_feat,
                                    n_legal, n_self_hidden)
        self.ally_net = EntityEncoder('ally', n_ally_hidden, n_attn_hidden)
        self.npc_net = EntityEncoder('npc', n_npc_feat, n_attn_hidden)
        self.enemy_net = EntityEncoder('enemy', n_enemy_feat, n_attn_hidden)

        self.interact_net = InteractionBlock(n_attn_hidden)
        self.lstm_net = MemoryBlock(n_attn_hidden, n_lstm_hidden)

        self.policy_head = PolicyHead(n_lstm_hidden, n_legal)
        self.value_head = nn.Linear(n_lstm_hidden, 1)

    @single_as_batch
    def infer(self, x, hx, cx):
        return self.forward(x, hx, cx, bptt_trunc_len=1)

    def forward(self, x, hx, cx, bptt_trunc_len):
        x = self._preprocess(x)

        h_self = self.self_net(x)
        h_ally = self.ally_net(self._self_as_ally_feature(h_self), h_self)
        h_npc = self.npc_net(x, h_self)
        h_enemy = self.enemy_net(x, h_self)

        h_inter = self.interact_net(x, h_self, h_ally, h_npc, h_enemy)
        h, c = self.lstm_net(x, h_self, h_inter, hx, cx, bptt_trunc_len)

        logits = self.policy_head(h)
        value = self.value_head(h.mean(dim=1))
        return logits, value, h, c

    @staticmethod
    def _preprocess(x):
        bs, na = x['tile'].shape[:2]
        team_mask_self = x['team_mask'][:, :, None]
        team_mask_ally = x['team_mask'].repeat(1, 2) \
            .unfold(dimension=1, size=na-1, step=1)[:, 1:-1]
        x['team_mask'] = torch.cat([team_mask_self, team_mask_ally], dim=-1)
        return x

    def _self_as_ally_feature(self, h_self):
        bs, na = h_self.shape[:2]
        tmp_x = dict()
        tmp_x['ally'] = h_self[:, :, :self.n_attn_hidden].repeat(1, 2, 1) \
            .unfold(dimension=1, size=na-1, step=1)[:, 1:-1].transpose(2, 3)
        return tmp_x
