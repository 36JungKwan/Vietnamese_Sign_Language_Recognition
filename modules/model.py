import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# [KHỐI CƠ SỞ 1] STANDARD ST-GCN BLOCK (Dùng cho Arch 1, 2, 4)
# ==============================================================================
class STGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes=76, temporal_kernel=9, stride=1):
        super().__init__()
        self.A = nn.Parameter(torch.ones((num_nodes, num_nodes)) / num_nodes, requires_grad=True)
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        padding = (temporal_kernel - 1) // 2
        self.temporal_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(temporal_kernel, 1), 
                                       stride=(stride, 1), padding=(padding, 0))
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = lambda x: x

    def forward(self, x):
        res = self.residual(x)
        x = self.spatial_conv(x) 
        x = torch.einsum('bctv,vw->bctw', x, self.A) 
        x = self.temporal_conv(x)
        x = self.bn(x)
        x += res
        return self.relu(x)

# ==============================================================================
# [KHỐI CƠ SỞ 2] CTR-GCN BLOCK (Dynamic Graph - Đồ thị động)
# ==============================================================================
class CTRGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes=76, temporal_kernel=9):
        super().__init__()
        # Đồ thị tĩnh (Học chung cho mọi frame)
        self.A_static = nn.Parameter(torch.ones((num_nodes, num_nodes)) / num_nodes, requires_grad=True)
        
        # Đồ thị động (Dynamic Topology - Tự sinh ra theo từng dáng tay)
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
        B, C, T, V = x.size()
        
        # 1. Tính toán Đồ thị động (Dynamic Adjacency Matrix) dựa trên đặc trưng hiện tại
        q = self.conv_q(x).mean(dim=2) # [B, C//4, V]
        k = self.conv_k(x).mean(dim=2) # [B, C//4, V]
        # Sinh ma trận A_dynamic shape [B, V, V]
        A_dynamic = torch.einsum('bcv,bcw->bvw', q, k) 
        A_dynamic = torch.softmax(A_dynamic, dim=-1)
        
        # Gộp đồ thị Tĩnh + Động
        x_spatial = self.spatial_conv(x) # [B, out_C, T, V]
        # Nhân với A_static (dùng chung) + A_dynamic (dùng riêng cho từng video)
        x_static = torch.einsum('bctv,vw->bctw', x_spatial, self.A_static)
        x_dynamic = torch.einsum('bctv,bvw->bctw', x_spatial, A_dynamic)
        x = x_static + x_dynamic
        
        # 2. Temporal Convolution
        x = self.temporal_conv(x)
        x = self.bn(x)
        return self.relu(x + res)


# ==============================================================================
# ARCHITECTURE 1: ST-GCN + TRANSFORMER (BASELINE)
# ==============================================================================
class STGCN_Transformer(nn.Module):
    def __init__(self, num_classes, in_channels=9, num_nodes=76, d_model=256, num_heads=8, num_layers=4, dropout=0.2):
        super().__init__()
        self.stgcn = nn.Sequential(STGCN_Block(in_channels, 64, num_nodes), STGCN_Block(64, d_model, num_nodes))
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, d_model)) 
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.stgcn(x).mean(dim=-1).permute(0, 2, 1) # -> [B, T, d_model]
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x).mean(dim=1)
        return self.classifier(x)


# ==============================================================================
# ARCHITECTURE 2: ST-GCN + BiLSTM (NHẸ, CHỐNG OVERFITTING TỐT)
# ==============================================================================
class STGCN_BiLSTM(nn.Module):
    def __init__(self, num_classes, in_channels=9, num_nodes=76, d_model=256, lstm_hidden_size=256, lstm_layers=2, dropout=0.4):
        super().__init__()
        self.stgcn = nn.Sequential(STGCN_Block(in_channels, 64, num_nodes), STGCN_Block(64, d_model, num_nodes))
        self.lstm = nn.LSTM(d_model, lstm_hidden_size, lstm_layers, batch_first=True, bidirectional=True, dropout=dropout if lstm_layers>1 else 0)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.stgcn(x).mean(dim=-1).permute(0, 2, 1) # -> [B, T, d_model]
        lstm_out, _ = self.lstm(x) 
        x = lstm_out.mean(dim=1) # -> [B, hidden_size*2]
        return self.classifier(x)


# ==============================================================================
# ARCHITECTURE 3: CTR-GCN (CHUYÊN GIA KHÔNG GIAN ĐỘNG - SPATIAL SOTA)
# ==============================================================================
class CTR_GCN_Model(nn.Module):
    def __init__(self, num_classes, in_channels=9, num_nodes=76, d_model=256):
        super().__init__()
        # Thay vì dùng ST-GCN, sử dụng hoàn toàn CTR-GCN block
        self.ctr_gcn = nn.Sequential(
            CTRGCN_Block(in_channels, 64, num_nodes),
            CTRGCN_Block(64, d_model, num_nodes),
            CTRGCN_Block(d_model, d_model, num_nodes)
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.ctr_gcn(x)         # [B, d_model, T, V]
        x = x.mean(dim=-1)          # Pooling không gian -> [B, d_model, T]
        x = x.mean(dim=-1)          # Pooling thời gian -> [B, d_model]
        return self.classifier(x)


# ==============================================================================
# ARCHITECTURE 4: ST-GCN + MS-TCN (CHUYÊN GIA THỜI GIAN - TEMPORAL SOTA)
# ==============================================================================
class STGCN_MSTCN(nn.Module):
    def __init__(self, num_classes, in_channels=9, num_nodes=76, d_model=256):
        super().__init__()
        # Trích xuất không gian bằng GCN
        self.stgcn = nn.Sequential(STGCN_Block(in_channels, 64, num_nodes), STGCN_Block(64, d_model, num_nodes))
        
        # Multi-Stage TCN (Dilated Convolutions)
        # Bắt các nhịp điệu thời gian dài ngắn khác nhau (Mở rộng trường nhìn)
        self.tcn_stage1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, dilation=1)
        self.tcn_stage2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2)
        self.tcn_stage3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=4, dilation=4)
        
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.stgcn(x).mean(dim=-1) # -> [B, d_model, T] (Phù hợp với Conv1d)
        
        # Chạy qua các màng lọc thời gian (TCN)
        res = x
        x = self.relu(self.tcn_stage1(x)) + res
        x = self.relu(self.tcn_stage2(x)) + x
        x = self.relu(self.tcn_stage3(x)) + x
        
        x = x.mean(dim=-1) # Pooling thời gian -> [B, d_model]
        return self.classifier(x)


# ==============================================================================
# ARCHITECTURE 5: SPATIAL-TEMPORAL TRANSFORMER (ST-TR - THUẦN ATTENTION)
# ==============================================================================
class ST_Transformer(nn.Module):
    def __init__(self, num_classes, in_channels=9, num_nodes=76, d_model=128, num_heads=4):
        super().__init__()
        self.node_embedding = nn.Linear(in_channels, d_model)
        
        # Spatial Transformer (Học liên kết giữa 76 khớp trong 1 frame)
        spatial_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.spatial_encoder = nn.TransformerEncoder(spatial_layer, num_layers=2)
        
        # Temporal Transformer (Học liên kết giữa các frame)
        temporal_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=2)
        
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Input: [B, C, T, V]
        B, C, T, V = x.size()
        x = x.permute(0, 2, 3, 1) # -> [B, T, V, C]
        
        # 1. Spatial Attention
        x = x.reshape(B*T, V, C)           # Trải phẳng Batch và Time
        x = self.node_embedding(x)         # -> [B*T, V, d_model]
        x = self.spatial_encoder(x)        # Tự chú ý giữa các khớp
        x = x.mean(dim=1)                  # Spatial Pooling -> [B*T, d_model]
        
        # 2. Temporal Attention
        x = x.reshape(B, T, -1)            # Phục hồi trục Time -> [B, T, d_model]
        x = self.temporal_encoder(x)       # Tự chú ý giữa các frame
        x = x.mean(dim=1)                  # Temporal Pooling -> [B, d_model]
        
        return self.classifier(x)


# ==============================================================================
# 4. MODEL FACTORY (TRẠM ĐIỀU HƯỚNG TỔNG)
# ==============================================================================
def create_model(cfg, num_classes):
    """ Tự động khởi tạo kiến trúc dựa trên config.yaml """
    model_name = cfg['experiment']['model_name'].lower()
    m_cfg = cfg['model'] # Trỏ tới block 'model' trong config
    
    # Lấy các tham số chung
    in_channels = m_cfg.get('in_channels', 9)
    num_nodes = m_cfg.get('num_nodes', 76)
    d_model = m_cfg.get('d_model', 256)
    dropout = m_cfg.get('dropout', 0.4)
    
    print(f"🚀 [MODEL FACTORY] Đang khởi tạo kiến trúc: {model_name.upper()}")
    
    if model_name == "stgcn_transformer":
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
            d_model=m_cfg.get('st_tr_d_model', 128), # Dùng d_model riêng chống tràn RAM
            num_heads=m_cfg.get('num_heads', 4)
        )
        
    else:
        raise ValueError(f"❌ Kiến trúc '{model_name}' chưa có trong danh sách Ngũ Hổ Tướng!")