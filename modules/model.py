import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# [KHỐI CƠ SỞ 1] STANDARD ST-GCN BLOCK
# ==============================================================================
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, stride=1):
        super(STGCNBlock, self).__init__()
        self.A = nn.Parameter(torch.randn(num_nodes, num_nodes))
        
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.temporal_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        A_norm = F.softmax(self.A, dim=1)
        x = torch.einsum('nctv,vw->nctw', x, A_norm)
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        x = self.bn(x)
        return self.relu(x + res)

# ==============================================================================
# [KHỐI CƠ SỞ 2] CTR-GCN BLOCK (Đồ thị động)
# ==============================================================================
class CTRGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes=76, temporal_kernel=9):
        super().__init__()
        self.A_static = nn.Parameter(torch.ones((num_nodes, num_nodes)) / num_nodes, requires_grad=True)
        
        self.conv_q = nn.Conv2d(in_channels, in_channels//4, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, in_channels//4, kernel_size=1)
        
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        padding = (temporal_kernel - 1) // 2
        self.temporal_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(temporal_kernel, 1), padding=(padding, 0))
        
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else lambda x: x

    def forward(self, x):
        res = self.residual(x)
        q = self.conv_q(x).mean(dim=2)
        k = self.conv_k(x).mean(dim=2)
        
        A_dynamic = torch.einsum('bcv,bcw->bvw', q, k)
        A_dynamic = torch.clamp(A_dynamic, min=-10.0, max=10.0) 
        A_dynamic = torch.softmax(A_dynamic, dim=-1)
        
        x_spatial = self.spatial_conv(x)
        x_static = torch.einsum('bctv,vw->bctw', x_spatial, self.A_static)
        x_dynamic = torch.einsum('bctv,bvw->bctw', x_spatial, A_dynamic)
        x = x_static + x_dynamic
        
        x = self.temporal_conv(x)
        x = self.bn(x)
        return self.relu(x + res)

# ==============================================================================
# [KHỐI CƠ SỞ 3] ARCFACE HEAD (Margin Loss) - BẢN FIX LỖI NAN
# ==============================================================================
class ArcFaceHead(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        super(ArcFaceHead, self).__init__()
        self.s = s
        self.m = m 
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, labels=None):
        # 1. Tắt AMP (autocast) cục bộ để ép khối lượng giác chạy thuần 100% bằng Float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            weight = self.weight.float()
            
            # Thêm eps=1e-5 để tuyệt đối ngăn chặn lỗi chia cho 0 (Division by Zero)
            cosine = F.linear(F.normalize(x, eps=1e-5), F.normalize(weight, eps=1e-5))
            
            if labels is None:
                return cosine * self.s

            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(1e-7, 1.0))
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
            one_hot = torch.zeros(cosine.size(), device=x.device, dtype=x.dtype)
            one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
            
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            return output * self.s


# ==============================================================================
# ARCHITECTURE V2: TWO-STREAM DECOUPLED GCN (SOTA - MỚI NHẤT)
# ==============================================================================
class TwoStreamVSLModel(nn.Module):
    def __init__(self, in_channels=9, num_classes=473, d_model=256, num_heads=8, num_layers=3):
        super(TwoStreamVSLModel, self).__init__()
        
        self.pose_gcn = nn.Sequential(
            STGCNBlock(in_channels, 64, num_nodes=33),
            STGCNBlock(64, 128, num_nodes=33)
        )
        self.hand_gcn = nn.Sequential(
            STGCNBlock(in_channels, 64, num_nodes=42),
            STGCNBlock(64, 128, num_nodes=42)
        )
        self.rel_mlp = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.fusion_linear = nn.Linear(384, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model*2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.arcface_head = ArcFaceHead(in_features=d_model, out_features=num_classes)

    def forward(self, x, labels=None):
        pose_x = x[:, :, :, :33]               
        left_hand_x = x[:, :, :, 33:54]        
        right_hand_x = x[:, :, :, 54:75]       
        chest_x = x[:, :, :, 75:76]            
        hand_x = torch.cat([left_hand_x, right_hand_x], dim=-1) 
        
        face_indices = [1, 2, 3, 4, 7, 8]
        face_centroid = pose_x[:, :3, :, face_indices].mean(dim=-1, keepdim=True) 
        chest_centroid = chest_x[:, :3, :, :]
        
        hand_coords = hand_x[:, :3, :, :] 
        hand_to_face = hand_coords - face_centroid
        hand_to_chest = hand_coords - chest_centroid
        
        rel_x = torch.cat([hand_to_face, hand_to_chest], dim=1) 
        
        feat_pose = self.pose_gcn(pose_x).mean(dim=-1) 
        feat_hand = self.hand_gcn(hand_x).mean(dim=-1) 
        feat_rel = self.rel_mlp(rel_x).mean(dim=-1)   
        
        feat_fused = torch.cat([feat_pose, feat_hand, feat_rel], dim=1) 
        feat_fused = feat_fused.permute(0, 2, 1) 
        feat_fused = self.fusion_linear(feat_fused) 
        
        T = feat_fused.size(1)
        feat_fused = feat_fused + self.pos_encoder[:, :T, :]
        out_seq = self.transformer(feat_fused) 
        
        out_features = out_seq.mean(dim=1) 
        return self.arcface_head(out_features, labels)


# ==============================================================================
# CÁC ARCHITECTURE CŨ (Thêm labels=None để tương thích ngược với file train.py mới)
# ==============================================================================
class STGCN_Transformer(nn.Module):
    def __init__(self, num_classes, in_channels=9, num_nodes=76, d_model=256, num_heads=8, num_layers=4, dropout=0.2):
        super().__init__()
        self.stgcn = nn.Sequential(STGCNBlock(in_channels, 64, num_nodes), STGCNBlock(64, d_model, num_nodes))
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, d_model)) 
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model*2, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, labels=None): # Đã fix
        x = self.stgcn(x).mean(dim=-1).permute(0, 2, 1)
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x).mean(dim=1)
        return self.classifier(x)

class STGCN_BiLSTM(nn.Module):
    def __init__(self, num_classes, in_channels=9, num_nodes=76, d_model=256, lstm_hidden_size=256, lstm_layers=2, dropout=0.4):
        super().__init__()
        self.stgcn = nn.Sequential(STGCNBlock(in_channels, 64, num_nodes), STGCNBlock(64, d_model, num_nodes))
        self.lstm = nn.LSTM(d_model, lstm_hidden_size, lstm_layers, batch_first=True, bidirectional=True, dropout=dropout if lstm_layers>1 else 0)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_classes)

    def forward(self, x, labels=None): # Đã fix
        x = self.stgcn(x).mean(dim=-1).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x) 
        x = lstm_out.mean(dim=1)
        return self.classifier(x)

class CTR_GCN_Model(nn.Module):
    def __init__(self, num_classes, in_channels=9, num_nodes=76, d_model=256):
        super().__init__()
        self.ctr_gcn = nn.Sequential(
            CTRGCN_Block(in_channels, 64, num_nodes),
            CTRGCN_Block(64, d_model, num_nodes),
            CTRGCN_Block(d_model, d_model, num_nodes)
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, labels=None): # Đã fix
        x = self.ctr_gcn(x).mean(dim=-1).mean(dim=-1)
        return self.classifier(x)

class STGCN_MSTCN(nn.Module):
    def __init__(self, num_classes, in_channels=9, num_nodes=76, d_model=256):
        super().__init__()
        self.stgcn = nn.Sequential(STGCNBlock(in_channels, 64, num_nodes), STGCNBlock(64, d_model, num_nodes))
        self.tcn_stage1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, dilation=1)
        self.tcn_stage2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2)
        self.tcn_stage3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=4, dilation=4)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, labels=None): # Đã fix
        x = self.stgcn(x).mean(dim=-1)
        res = x
        x = self.relu(self.tcn_stage1(x)) + res
        x = self.relu(self.tcn_stage2(x)) + x
        x = self.relu(self.tcn_stage3(x)) + x
        x = x.mean(dim=-1)
        return self.classifier(x)

class ST_Transformer(nn.Module):
    def __init__(self, num_classes, in_channels=9, num_nodes=76, d_model=128, num_heads=4):
        super().__init__()
        self.node_embedding = nn.Linear(in_channels, d_model)
        self.spatial_pe = nn.Parameter(torch.randn(1, num_nodes, d_model)) 
        self.temporal_pe = nn.Parameter(torch.randn(1, 100, d_model)) 
        
        spatial_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.spatial_encoder = nn.TransformerEncoder(spatial_layer, num_layers=2)
        temporal_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=2)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, labels=None): # Đã fix
        B, C, T, V = x.size()
        x = x.permute(0, 2, 3, 1)
        
        x = x.reshape(B*T, V, C)
        x = self.node_embedding(x)
        x = x + self.spatial_pe[:, :V, :]
        x = x + self.spatial_encoder(x)
        x = x.mean(dim=1)
        
        x = x.reshape(B, T, -1)
        x = x + self.temporal_pe[:, :T, :]
        x = x + self.temporal_encoder(x)
        x = x.mean(dim=1)
        
        return self.classifier(x)


# ==============================================================================
# 4. MODEL FACTORY (TRẠM ĐIỀU HƯỚNG TỔNG)
# ==============================================================================
def create_model(cfg, num_classes):
    """ Tự động khởi tạo kiến trúc dựa trên config.yaml """
    model_name = cfg['experiment']['model_name'].lower()
    m_cfg = cfg['model'] 
    
    in_channels = m_cfg.get('in_channels', 9)
    num_nodes = m_cfg.get('num_nodes', 76)
    d_model = m_cfg.get('d_model', 256)
    dropout = m_cfg.get('dropout', 0.4)
    
    print(f"🚀 [MODEL FACTORY] Đang khởi tạo kiến trúc: {model_name.upper()}")
    
    # 🌟 Đã thêm kiến trúc Two-Stream mới vào Trạm điều hướng
    if model_name == "two_stream_vsl":
        return TwoStreamVSLModel(
            in_channels=in_channels,
            num_classes=num_classes,
            d_model=d_model,
            num_heads=m_cfg.get('num_heads', 8),
            num_layers=m_cfg.get('num_layers', 3)
        )
        
    elif model_name == "stgcn_transformer":
        return STGCN_Transformer(
            num_classes=num_classes, 
            in_channels=in_channels, 
            num_nodes=num_nodes, 
            d_model=d_model, 
            num_heads=m_cfg.get('num_heads', 8), 
            num_layers=m_cfg.get('num_layers', 4),
            dropout=dropout
        )
        
    elif model_name == "stgcn_bilstm":
        return STGCN_BiLSTM(
            num_classes=num_classes, 
            in_channels=in_channels, 
            num_nodes=num_nodes, 
            d_model=d_model,
            lstm_hidden_size=m_cfg.get('lstm_hidden_size', 256),
            lstm_layers=m_cfg.get('lstm_layers', 2),
            dropout=dropout
        )
        
    elif model_name == "ctr_gcn":
        return CTR_GCN_Model(num_classes, in_channels, num_nodes, d_model)
        
    elif model_name == "stgcn_mstcn":
        return STGCN_MSTCN(num_classes, in_channels, num_nodes, d_model)
        
    elif model_name == "st_tr":
        return ST_Transformer(
            num_classes=num_classes, 
            in_channels=in_channels, 
            num_nodes=num_nodes, 
            d_model=m_cfg.get('st_tr_d_model', 128),
            num_heads=m_cfg.get('num_heads', 4)
        )
        
    else:
        raise ValueError(f"❌ Kiến trúc '{model_name}' chưa có trong danh sách hỗ trợ!")