
class TinyGate(nn.Module):
    def __init__(self, channels, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = max(8, channels // 4)
        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, channels, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, _, _ = x.shape
        y = F.adaptive_avg_pool2d(x, 1).view(B, C)          # (B, C)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.sig(y).view(B, C, 1, 1)
        return x * y


import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossLayerChannelFusion(nn.Module):
    def __init__(self, channels, expansion_factor=2):
        super().__init__()

        self.channels = channels
        hidden_dim = channels * expansion_factor

        self.gated_mlp = nn.Sequential(
            nn.LayerNorm(channels, eps=1e-6),
   
            nn.Linear(channels, hidden_dim * 2) 
        )
        self.projection = nn.Linear(hidden_dim, channels)

        # Giữ nguyên
        self.control_mlp = nn.Sequential(
            nn.Linear(channels, max(16, channels // 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(16, channels // 4), channels)
        )

    def forward(self, x):

        y = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        
        gated_output = self.gated_mlp(y)
        data, gate = gated_output.chunk(2, dim=-1)
        

        mixed_channels = data * F.sigmoid(gate) # Shape: (B, H, W, hidden_dim)   

        projected_mixed = self.projection(mixed_channels) # Shape: (B, H, W, C)

        control_vector_input = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        control_signal = self.control_mlp(control_vector_input)
        control_signal = control_signal.unsqueeze(1).unsqueeze(1) # Shape: (B, 1, 1, C)

        fused = projected_mixed * control_signal
        
        out = fused.permute(0, 3, 1, 2) 
        
        return out


class Gate_Spatial_Channel__Unit(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.C = channels

        self.channel_branch = CrossLayerChannelFusion(channels) 

   
        self.dw1x3 = nn.Conv2d(self.C, self.C, kernel_size=(1,3), padding=(0,1), groups=self.C, bias=False)
        self.dw3x1 = nn.Conv2d(self.C, self.C, kernel_size=(3,1), padding=(1,0), groups=self.C, bias=False)
        self.dw3x3 = nn.Conv2d(self.C, self.C, kernel_size=3, padding=1, groups=self.C, bias=False)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.gate_fc = nn.Sequential(
            nn.Conv2d(self.C, max(4, self.C // 4), kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(4, self.C // 4), 3, kernel_size=1),
            nn.Sigmoid()
        )

        self.pw = nn.Conv2d(self.C, self.C, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(self.C)
        self.relu = nn.ReLU(inplace=True)
 
        self.gate = TinyGate(self.C, hidden=max(8, self.C // 2))

    def forward(self, x):
        identity = x
        
        
        channel_feat = self.channel_branch(x)

        b1 = self.dw1x3(x)
        b2 = self.dw3x1(x)
        b3 = self.dw3x3(x)

        gates = self.gate_fc(self.global_pool(x))
        g1, g2, g3 = gates[:, 0:1], gates[:, 1:2], gates[:, 2:3]

        sp = g1 * b1 + g2 * b2 + g3 * b3
        sp = self.pw(sp)
        sp = self.bn(sp)
        sp = self.relu(sp)

        fused = channel_feat + sp 
        gated = self.gate(fused)
        out = identity + gated
        return out
    

class FCSA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.C = channels

        def ds_conv(ch):
            # depthwise then pointwise (cheap)
            return nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=False),
                nn.Conv2d(ch, ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True)
            )

        self.proc0 = ds_conv(self.C)  # full
        self.proc1 = ds_conv(self.C)  # half
        self.proc2 = ds_conv(self.C)  # quarter

        # per-scale gating (TinyGate)
        self.g0 = TinyGate(self.C, hidden=max(8, self.C // 2))
        self.g1 = TinyGate(self.C, hidden=max(8, self.C // 2))
        self.g2 = TinyGate(self.C, hidden=max(8, self.C // 2))

        # small 1x1 convs for up/down fusion
        self.up1 = nn.Conv2d(self.C, self.C, kernel_size=1, bias=False)
        self.up2 = nn.Conv2d(self.C, self.C, kernel_size=1, bias=False)
        self.down1 = nn.Conv2d(self.C, self.C, kernel_size=1, bias=False)
        self.down2 = nn.Conv2d(self.C, self.C, kernel_size=1, bias=False)

        # scale-MLPs: GAP -> small MLP -> sigmoid (adds params but tiny FLOPs)
        hidden_scale = max(16, self.C)
        self.scale_mlp0 = nn.Sequential(nn.Linear(self.C, hidden_scale, bias=False),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_scale, self.C, bias=False),
                                        nn.Sigmoid())
        self.scale_mlp1 = nn.Sequential(nn.Linear(self.C, hidden_scale, bias=False),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_scale, self.C, bias=False),
                                        nn.Sigmoid())
        self.scale_mlp2 = nn.Sequential(nn.Linear(self.C, hidden_scale, bias=False),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_scale, self.C, bias=False),
                                        nn.Sigmoid())

        # final projection
        self.final = nn.Conv2d(self.C, self.C, kernel_size=1, bias=False)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # scales (use adaptive pools so any H/W works)
        s0 = x
        s1 = F.adaptive_avg_pool2d(x, (max(1, H // 2), max(1, W // 2)))
        s2 = F.adaptive_avg_pool2d(x, (max(1, H // 4), max(1, W // 4)))

        # per-scale processing
        p0 = self.proc0(s0)
        p1 = self.proc1(s1)
        p2 = self.proc2(s2)

        # coarse -> fine (upsample coarse)
        p2_u = F.interpolate(p2, size=p1.shape[2:], mode='nearest')
        p2_u = self.up2(p2_u)
        p1_comb = p1 + p2_u

        p1_u = F.interpolate(p1_comb, size=p0.shape[2:], mode='nearest')
        p1_u = self.up1(p1_u)
        p0_comb = p0 + p1_u

        # fine -> coarse feedback (downsample fine)
        p0_d = F.adaptive_avg_pool2d(p0, p1.shape[2:])
        p0_d = self.down1(p0_d)
        p1_comb2 = p1_comb + p0_d

        p1_d = F.adaptive_avg_pool2d(p1_comb2, p2.shape[2:])
        p1_d = self.down2(p1_d)
        p2_comb = p2 + p1_d

        # per-scale gating
        p0_g = self.g0(p0_comb)
        p1_g = self.g1(p1_comb2)
        p2_g = self.g2(p2_comb)

        # scale-MLP weights (computed from GAP descriptors)
        d0 = F.adaptive_avg_pool2d(p0_g, 1).view(B, C)
        d1 = F.adaptive_avg_pool2d(p1_g, 1).view(B, C)
        d2 = F.adaptive_avg_pool2d(p2_g, 1).view(B, C)
        m0 = self.scale_mlp0(d0).view(B, C, 1, 1)
        m1 = self.scale_mlp1(d1).view(B, C, 1, 1)
        m2 = self.scale_mlp2(d2).view(B, C, 1, 1)

        p0_g = p0_g * m0
        p1_g = p1_g * m1
        p2_g = p2_g * m2

        # bring all to full res and sum
        p1_up = F.interpolate(p1_g, size=p0_g.shape[2:], mode='nearest')
        p2_up = F.interpolate(p2_g, size=p0_g.shape[2:], mode='nearest')
        out = p0_g + p1_up + p2_up

        out = self.final(out)
        out = x + out
        return out


class SynerNet(nn.Module):
    def __init__(self, channels_per_stage= (8, 8, 16, 16, 32, 32), num_classes=12):
        super().__init__()
        assert len(channels_per_stage) == 6, "channels_per_stage must be length 6"
        self.chs = list(channels_per_stage)
        c0 = self.chs[0]

        # stem: early downsample -> 224 -> 112 -> 56: reduces FLOPs heavily downstream
        self.stem = nn.Sequential(
            nn.Conv2d(3, c0, kernel_size=3, stride=2, padding=1, bias=False),  # 224 -> 112
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
            nn.Conv2d(c0, c0, kernel_size=3, stride=2, padding=1, bias=False),  # 112 -> 56
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
        )

        # build alternating blocks
        self.blocks = nn.ModuleList()
        for i, ch in enumerate(self.chs):
            proj = None
            if i == 0:
                in_ch = c0
            else:
                in_ch = self.chs[i - 1]
            if in_ch != ch:
                proj = nn.Conv2d(in_ch, ch, kernel_size=1, bias=False)
            module = Gate_Spatial_Channel__Unit (ch) if (i % 2 == 0) else FCSA(ch)
            self.blocks.append(nn.ModuleDict({"proj": proj, "module": module}))

        # final pooling + classifier 
        final_ch = self.chs[-1]
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_ch, max(128, final_ch * 4)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(max(128, final_ch * 4), num_classes)
        )

    def forward(self, x):
        x = self.stem(x)    
        for blk in self.blocks:
            if blk["proj"] is not None:
                x = blk["proj"](x)
            x = blk["module"](x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.classifier(x)

channels = (8, 8, 16, 16, 32, 32)   
model = SynerNet(channels_per_stage=channels, num_classes=12)
