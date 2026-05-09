# 🔬 Phân Tích Chi Tiết Từng Layer - YOLOv8 GoldACsim

## 📋 Mục Lục
1. [BACKBONE Layers](#backbone-layers)
2. [HEAD Layers](#head-layers)
3. [Detection Head](#detection-head)
4. [Tính Toán Math](#tính-toán-math)

---

## BACKBONE LAYERS

### **Layer 0: Conv (3D Input) → 320×320×64**

#### Code:
```python
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Mish()  # activation

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# Layer 0: Conv, [64, 3, 2]
# Meaning: out_channels=64, kernel=3×3, stride=2
```

#### Tính Toán:
```
Input:   1 × 3 × 640 × 640
         ↓ Conv(k=3, s=2, padding=1)
         ↓ BatchNorm
         ↓ Mish activation
Output:  1 × 64 × 320 × 320

Formula cho output size:
  H_out = ⌊(H_in + 2×padding - dilation×(kernel-1) - 1) / stride⌋ + 1
  H_out = ⌊(640 + 2×1 - 1×(3-1) - 1) / 2⌋ + 1 = ⌊640 / 2⌋ = 320
  W_out = 320 (tương tự)
```

#### Công Dụng:
- **Downsampling** từ 640→320 (stride=2)
- **Feature extraction** với kernel 3×3
- **Output channels** từ 3→64
- **BatchNorm** chuẩn hóa dữ liệu
- **Mish** activation: y = x × tanh(softplus(x)) - non-linear hơn ReLU

---

### **Layer 1: Conv → 160×160×128**

```python
# Layer 1: Conv, [128, 3, 2]
Input:   1 × 64 × 320 × 320
         ↓ Conv(64→128, k=3, s=2)
Output:  1 × 128 × 160 × 160
```

#### Tính Toán:
```
Output kích thước:
  H_out = ⌊(320 + 2×1 - 2) / 2⌋ + 1 = 160
  W_out = 160
```

---

### **Layer 2: C2f → 160×160×128** (P2 mức độ chi tiết)

#### Code:
```python
class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels = 128 × 0.5 = 64
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # 128 → 128 (1×1 conv)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # (2+n)×64 → 128
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3,3),(3,3)), e=1.0) 
            for _ in range(n)  # n=3 Bottleneck blocks
        )
    
    def forward(self, x):
        """Forward pass through C2f layer."""
        # Bước 1: Tách input thành 2 nhánh
        y = list(self.cv1(x).chunk(2, 1))  # x: 160×160×128 → 2×(160×160×64)
        
        # Bước 2: Xử lý qua n Bottleneck blocks
        # y = [branch1, branch2]
        # Sau cv1: y = [64, 64] channels (tách đôi)
        y.extend(m(y[-1]) for m in self.m)  # Thêm 3 output từ Bottleneck
        # y = [branch1, bottleneck1, bottleneck2, bottleneck3]
        
        # Bước 3: Gộp tất cả lại
        # Total: 64 + 64 + 64 + 64 = (2+3)×64 = 5×64 = 320
        return self.cv2(torch.cat(y, 1))  # 320 → 128
```

#### Bottleneck Class:
```python
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3,3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # 64 × 0.5 = 32
        self.cv1 = Conv(c1, c_, k[0], 1)  # 64 → 32 (3×3)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)  # 32 → 64 (3×3)
        self.add = shortcut and c1 == c2  # Shortcut nếu c1==c2

    def forward(self, x):
        # Residual connection
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
```

#### Tính Toán Chi Tiết:
```
Input:  1 × 128 × 160 × 160

Bước 1 - cv1 (Conv 1×1):
  128 → 128 channels (2 × 64)
  Output: 1 × 128 × 160 × 160
  chunk(2, 1): Tách thành 2 nhánh × 64 channels
  → [1×64×160×160, 1×64×160×160]

Bước 2 - Bottleneck layers (n=3):
  Mỗi Bottleneck:
    - Conv(64→32, 3×3): 32 channels
    - Conv(32→64, 3×3): 64 channels
    - Shortcut: x + output (vì c1=c2=64)
  
  Bottleneck 1: input y[-1]=branch2 → output 1×64×160×160
  Bottleneck 2: input Bottleneck1_out → output 1×64×160×160
  Bottleneck 3: input Bottleneck2_out → output 1×64×160×160
  
  y = [branch1(64), branch2(64), bn1(64), bn2(64), bn3(64)]
     = 5 × 64 channels

Bước 3 - cv2 (Conv 1×1):
  (2+3)×64=320 → 128 channels
  Output: 1 × 128 × 160 × 160

✅ Final Output: 160 × 160 × 128 (nhỏ gấp 4 so với input)
```

#### Công Dụng:
- **CSP Bottleneck** - Cross Stage Partial Network
- Tăng **receptive field** từ 3 Bottleneck blocks
- **Shortcut connections** cải thiện gradient flow
- Output vẫn giữ kích thước nhưng **tăng độ sâu đặc trưng**

---

### **Layer 3: Conv → 80×80×256**

```python
# Layer 3: Conv, [256, 3, 2]
Input:   1 × 128 × 160 × 160
         ↓ Conv(128→256, k=3, s=2)
Output:  1 × 256 × 80 × 80
```

---

### **Layer 4: C2f → 80×80×256** (P3 - mức độ chi tiết cao)

```python
# Tương tự Layer 2 nhưng với n=6 Bottleneck blocks
Input:   1 × 256 × 80 × 80
         ↓ cv1: 256 → 256 (2×128)
         ↓ 6× Bottleneck(128→128)
         ↓ cv2: (2+6)×128 → 256
Output:  1 × 256 × 80 × 80
```

#### Ưu Điểm Layer 4 vs Layer 2:
- **6 Bottleneck blocks** (vs 3 ở Layer 2) → **receptive field lớn hơn**
- Bắt được **pattern phức tạp hơn**
- **P3 (80×80)** = mức độ chi tiết cao cho phát hiện **đối tượng nhỏ**

---

### **Layer 5: Conv → 40×40×512**

```python
# Layer 5: Conv, [512, 3, 2]
Input:   1 × 256 × 80 × 80
         ↓ Conv(256→512, k=3, s=2)
Output:  1 × 512 × 40 × 40
```

---

### **Layer 6: C2f → 40×40×512** (P4 - mức độ chi tiết trung bình)

```python
# Tương tự Layer 4 nhưng với n=6 Bottleneck blocks
Input:   1 × 512 × 40 × 40
         ↓ cv1: 512 → 512 (2×256)
         ↓ 6× Bottleneck(256→256)
         ↓ cv2: (2+6)×256 → 512
Output:  1 × 512 × 40 × 40
```

---

### **Layer 7: Conv → 20×20×1024**

```python
# Layer 7: Conv, [1024, 3, 2]
Input:   1 × 512 × 40 × 40
         ↓ Conv(512→1024, k=3, s=2)
Output:  1 × 1024 × 20 × 20
```

---

### **Layer 8: C2f → 20×20×1024**

```python
# Layer 8: C2f, [1024, 3]
Input:   1 × 1024 × 20 × 20
         ↓ cv1: 1024 → 1024 (2×512)
         ↓ 3× Bottleneck(512→512)
         ↓ cv2: (2+3)×512 → 1024
Output:  1 × 1024 × 20 × 20
```

---

### **Layer 9: SimSPPF → 20×20×1024** (P5 - mức độ chi tiết thấp)

#### Code:
```python
class SimSPPF(nn.Module):
    '''Simplified SPPF with ReLU activation'''
    
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels = 1024 // 2 = 512
        self.cv1 = SimConv(c1, c_, 1, 1)  # 1024 → 512 (1×1)
        self.cv2 = SimConv(c_ * 4, c2, 1, 1)  # 512×4 → 1024 (1×1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)  # k=5, padding=2

    def forward(self, x):
        x = self.cv1(x)  # 1024 → 512
        
        # Spatial Pyramid Pooling - gộp 4 mức độ
        y1 = x  # Original: 512
        y2 = self.m(x)  # MaxPool 5×5: 512
        y3 = self.m(y2)  # MaxPool 5×5 lần 2: 512
        y4 = self.m(y3)  # MaxPool 5×5 lần 3: 512
        
        # Gộp tất cả: 512 + 512 + 512 + 512 = 2048
        return self.cv2(torch.cat([y1, y2, y3, y4], 1))  # 2048 → 1024
```

#### Tính Toán MaxPool:
```
Input: 1 × 512 × 20 × 20
MaxPool(k=5, s=1, p=2):
  H_out = ⌊(20 + 2×2 - 5) / 1⌋ + 1 = ⌊19 / 1⌋ + 1 = 20
  Output: 1 × 512 × 20 × 20

(Kích thước không đổi, nhưng features được gộp từ 5×5 neighborhood)
```

#### Công Dụng SimSPPF:
```
┌─────────────────────────────────────────┐
│ Input: 20×20×1024                       │
└─────────────────────────────────────────┘
         │
         ├─ cv1: 1024→512
         │
    ┌────┴─────────┐
    │ Pyramid Pool │
    ├────┬────┬────┤
    │ y1 │ y2 │ y3 │ y4
    │ │  │ │  │ │  │ │
    │ 5× │ 5× │ 5× │ (nếu y3 có kích thước 20×20)
    └────┴────┴────┘
         │
      Concatenate: [512, 512, 512, 512] → 2048 channels
         │
      cv2: 2048 → 1024
         │
    Output: 20×20×1024

✅ Ưu điểm:
- Gộp thông tin từ MULTIPLE RECEPTIVE FIELDS
- y1 = local details (5×5 neighborhood không gộp)
- y2, y3, y4 = toàn bộ khu vực (gộp tối đa)
- Bắt được đặc trưng ở MỌI quy mô
```

**Mục đích:** Capture features ở multiple receptive fields để phát hiện đối tượng lớn

---

## HEAD LAYERS

### **Layer 10: SimFusion_4in → Fuse [P2, P3, P4, P5]**

#### Code:
```python
class SimFusion_4in(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d

    def forward(self, x):
        x_l, x_m, x_s, x_n = x  # 4 inputs
        # x_l = L2 (P2) = 160×160×128
        # x_m = L4 (P3) = 80×80×256
        # x_s = L6 (P4) = 40×40×512
        # x_n = L9 (P5) = 20×20×1024
        
        B, C, H, W = x_s.shape  # 40×40
        output_size = np.array([H, W])  # [40, 40]
        
        # Downsample tất cả về cùng kích thước (40×40)
        x_l = self.avg_pool(x_l, output_size)  # 160×160→40×40 (adaptive avg pool)
        x_m = self.avg_pool(x_m, output_size)  # 80×80→40×40 (adaptive avg pool)
        # x_s đã là 40×40
        x_n = F.interpolate(x_n, size=(H, W), mode='bilinear', align_corners=False)  # 20×20→40×40
        
        # Gộp lại
        out = torch.cat([x_l, x_m, x_s, x_n], 1)  # Concatenate theo channel
        # Channels: 128 + 256 + 512 + 1024 = 1920
        return out  # Output: 40×40×1920
```

#### Tính Toán AdaptiveAvgPool:
```
AdaptiveAvgPool(input, output_size):
  Công dụng: Gộp lại input về output_size một cách automatic
  
  Input: 160×160×128, output_size: 40×40
  - Chia input thành 40×40 regions (mỗi region 4×4)
  - Lấy trung bình của mỗi region
  → Output: 40×40×128
  
  Input: 80×80×256, output_size: 40×40
  - Chia input thành 40×40 regions (mỗi region 2×2)
  - Lấy trung bình của mỗi region
  → Output: 40×40×256
```

#### Công Dụng:
- **Multi-scale fusion** - kết hợp thông tin từ 4 cấp độ khác nhau
- **Tất cả đưa về 40×40** - kích thước trung bình
- **Information pooling** - tập hợp thông tin từ mọi độ phân giải

---

### **Layer 11: IFM → Intermediate Feature Module**

#### Code:
```python
class IFM(nn.Module):
    def __init__(self, inc, ouc, embed_dim_p=96, fuse_block_num=3):
        super().__init__()
        # inc = 1920 (từ SimFusion_4in)
        # ouc = [64, 32] (tách thành 2 phần)
        # embed_dim_p = 96
        # fuse_block_num = 3
        
        self.conv = nn.Sequential(
            Conv(inc, embed_dim_p),  # 1920 → 96
            *[RepVGGBlock(embed_dim_p, embed_dim_p) for _ in range(fuse_block_num)],  # 3× RepVGG
            Conv(embed_dim_p, sum(ouc))  # 96 → 64+32 = 96
        )
    
    def forward(self, x):
        # x: 40×40×1920 (từ SimFusion_4in)
        # 1920 → 96 (Conv)
        # 96 → 96 (RepVGG 3×)
        # 96 → 96 (Conv)
        return self.conv(x)  # Output: 40×40×96 → split thành [64, 32]
```

#### RepVGGBlock:
```python
# RepVGG = Reparameterized VGG
# Kết hợp multiple branches để tăng cường feature learning
# Tương tự RepConv - có Conv 3×3 + Conv 1×1 + identity
```

#### Công Dụng:
- **Tạo đặc trưng trung gian** để chia sẻ giữa các nhánh
- **96 channels** được tách thành **[64, 32]** cho 2 nhánh khác nhau
- **RepVGGBlock** cải thiện học đặc trưng thông qua reparameterization

---

### **Layer 12: Conv 1×1 → 512 channels**

```python
# Layer 12: Conv, [512, 1, 1]
# Input: L9 (P5) = 20×20×1024
Input:   1 × 1024 × 20 × 20
         ↓ Conv(k=1, s=1) - không thay đổi kích thước
Output:  1 × 512 × 20 × 20

Công dụng: 
- Giảm channels từ 1024 → 512
- Tính toán nhẹ (chỉ 1×1)
```

---

### **Layer 13: SimFusion_3in → Fuse [P3, P4, Conv(L12)]**

#### Code:
```python
class SimFusion_3in(nn.Module):
    def __init__(self, in_channel_list, out_channels):
        super().__init__()
        # in_channel_list = [256, 512, 512] (L4, L6, L12)
        # out_channels = 512
        
        # Chuẩn hóa channels về out_channels
        self.cv1 = Conv(256, 512, act=nn.ReLU()) if 256 != 512 else nn.Identity()
        self.cv2 = Conv(512, 512, act=nn.ReLU()) if 512 != 512 else nn.Identity()  # Identity
        self.cv3 = Conv(512, 512, act=nn.ReLU()) if 512 != 512 else nn.Identity()  # Identity
        
        # Fusion layer
        self.cv_fuse = Conv(512 * 3, 512, act=nn.ReLU())
        self.downsample = nn.functional.adaptive_avg_pool2d

    def forward(self, x):  # x = [L4, L6, L12]
        # x[0] = L4 (P3) = 80×80×256
        # x[1] = L6 (P4) = 40×40×512
        # x[2] = L12 = 20×20×512
        
        N, C, H, W = x[1].shape  # 40×40 (lấy từ x[1])
        output_size = (H, W)  # [40, 40]
        
        # Đưa tất cả về 40×40
        x0 = self.cv1(self.downsample(x[0], output_size))  # 80×80 → 40×40 → Conv
        x1 = self.cv2(x[1])  # 40×40 → Identity (vì 512==512)
        x2 = self.cv3(F.interpolate(x[2], size=(H, W), mode='bilinear'))  # 20×20 → 40×40 → Conv
        
        # Gộp lại
        return self.cv_fuse(torch.cat((x0, x1, x2), dim=1))  # [512, 512, 512] → 1536 → Conv → 512
```

#### Tính Toán:
```
Đầu vào:
  x[0] (L4): 80×80×256   (P3 - mức độ chi tiết cao)
  x[1] (L6): 40×40×512   (P4 - mức độ chi tiết trung bình)
  x[2] (L12): 20×20×512  (từ L9 rút gọn)

Đưa về 40×40:
  x[0]: AdaptiveAvgPool(80×80→40×40) → 40×40×256 → Conv(256→512) → 40×40×512
  x[1]: 40×40×512 → Identity → 40×40×512
  x[2]: Interpolate(20×20→40×40) → 40×40×512 → Conv(512→512) → 40×40×512

Concatenate:
  [40×40×512, 40×40×512, 40×40×512] → 40×40×1536

Fusion:
  Conv(1536 → 512) → 40×40×512

✅ Mục đích: Kết hợp 3 mức độ chi tiết khác nhau về 40×40
```

---

### **Layer 14: InjectionMultiSum_Auto_pool**

#### Code:
```python
class InjectionMultiSum_Auto_pool(nn.Module):
    def __init__(self, inp, oup, global_inp, flag):
        super().__init__()
        self.global_inp = global_inp  # [64, 32]
        self.flag = flag  # 0 hoặc 1
        self.local_embedding = Conv(inp, oup, 1, act=False)  # Local path
        self.global_embedding = Conv(global_inp[flag], oup, 1, act=False)  # Global path
        self.global_act = Conv(global_inp[flag], oup, 1, act=False)  # Attention path
        self.act = h_sigmoid()  # Hard sigmoid (0-1)

    def forward(self, x):
        '''x = [local_features, global_features]'''
        x_l, x_g = x
        B, C, H, W = x_l.shape
        g_B, g_C, g_H, g_W = x_g.shape
        
        # Chọn nhánh global dựa trên flag
        global_info = x_g.split(self.global_inp, dim=1)[self.flag]
        
        # Local path
        local_feat = self.local_embedding(x_l)  # Chuẩn hóa channels
        
        # Global path + Attention
        global_act = self.global_act(global_info)  # Tính attention weights
        global_feat = self.global_embedding(global_info)  # Tính global features
        
        # Nếu kích thước global lớn hơn local → pool xuống
        if H < g_H:  # Cần giảm size
            avg_pool = get_avg_pool()
            output_size = [H, W]
            sig_act = avg_pool(global_act, output_size)  # Global attention pool
            global_feat = avg_pool(global_feat, output_size)  # Global features pool
        else:  # Cần tăng size
            sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear')
            global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear')
        
        # Element-wise multiplication + addition
        out = local_feat * sig_act + global_feat
        return out
```

#### Công Dụng (flag=0 for Layer 14):
```
Layer 14: InjectionMultiSum_Auto_pool([512, [64, 32], 0])

Đầu vào:
  - x_l (local): 40×40×512 (từ Layer 13)
  - x_g (global): [64, 32] từ Layer 11 IFM

Xử lý:
  - Chọn branch 0 → 64 channels
  - Local embedding: 512 → 512 (Conv 1×1)
  - Global embedding: 64 → 512 (Conv 1×1)
  - Global attention: 64 → 512 (Conv 1×1) → h_sigmoid → [0,1]
  
Công thức:
  output = local_feat × global_attention + global_feat
  
  Ý nghĩa:
    - global_attention = soft mask (0-1) điều chỉnh global features
    - local_feat × attention = chọn local features quan trọng
    - + global_feat = thêm thông tin global
  
✅ Mục đích: TIÊM thông tin toàn cục vào features cục bộ
```

---

### **Layer 15: C3Ghost → 40×40×512**

#### Code:
```python
class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""
    
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))

class GhostBottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2  # 512 // 2 = 256
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # c1 → 256
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),
            GhostConv(c_, c2, 1, 1, act=False),  # 256 → c2
        )
        self.shortcut = ... # Skip connection

class GhostConv(nn.Module):
    """Ghost Convolution - tạo features giả từ linear transformation"""
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)  # Primary features
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)  # Cheap features (DWConv 5×5)

    def forward(self, x):
        y = self.cv1(x)  # Tính features chính
        return torch.cat((y, self.cv2(y)), 1)  # Gộp: [primary, cheap_generated]
```

#### Công Dụng Ghost Convolution:
```
┌─────────────────┐
│ Input: 512 ch   │
└─────────────────┘
        │
        ├─ Primary Conv (50%)
        │   └─ 256 channels (tính toán hầu hết)
        │
        └─ Cheap Generation (50%)
            └─ 5×5 DWConv trên primary features
                └─ 256 channels (tính toán ít)

Output: [256_primary, 256_cheap] = 512 channels

✅ Ưu điểm:
- Giảm 50% tính toán
- Vẫn giữ độ chính xác cao
- Dùng cho mobile/edge devices
```

---

### **Layer 16: Conv 1×1 → 256 channels**

```python
# Layer 16: Conv, [256, 1, 1]
Input:   1 × 512 × 40 × 40 (L6 = P4)
         ↓ Conv(512→256, k=1)
Output:  1 × 256 × 40 × 40
```

---

### **Layer 17: SimFusion_3in → Fuse [P2, P3, Conv(L16)]**

```
Đầu vào:
  - L2 (P2) = 160×160×128
  - L4 (P3) = 80×80×256
  - L16 = 40×40×256

Đưa về 40×40:
  - L2: 160×160 → 40×40 (AdaptiveAvgPool, chia 4×4)
  - L4: 80×80 → 40×40 (AdaptiveAvgPool, chia 2×2)
  - L16: 40×40 → 40×40 (không thay đổi)

Concatenate + Fusion:
  Channels: 128 + 256 + 256 = 640
  → Conv(640→256) → 40×40×256
```

---

### **Layer 18-19: InjectionMultiSum + C3Ghost**

```
Tương tự Layer 14-15 nhưng:
- Layer 18: InjectionMultiSum_Auto_pool([256, [64, 32], 1])
  Chọn branch 1 → 32 channels từ IFM

- Layer 19: C3Ghost(256→256)
  Output: 80×80×256 (được interpolate từ 40×40)
```

---

### **Layer 20: PyramidPoolAgg → Aggregate 3 scales**

#### Code:
```python
class PyramidPoolAgg(nn.Module):
    def __init__(self, inc, ouc, stride, pool_mode='torch'):
        super().__init__()
        self.stride = stride  # stride = 2
        self.pool = nn.functional.adaptive_avg_pool2d
        self.conv = Conv(inc, ouc)  # inc = 3 scales gộp lại

    def forward(self, inputs):  # inputs = [L19, L15, L9]
        # L19: 80×80×256
        # L15: 40×40×512
        # L9: 20×20×1024
        
        B, C, H, W = get_shape(inputs[-1])  # 20×20
        H = (H - 1) // self.stride + 1  # (20-1)//2 + 1 = 10
        W = (W - 1) // self.stride + 1  # (20-1)//2 + 1 = 10
        
        output_size = [10, 10]
        
        # Pool all inputs về 10×10
        out = [self.pool(inp, output_size) for inp in inputs]
        # out[0]: AdaptiveAvgPool(80×80→10×10, k=8) → 10×10×256
        # out[1]: AdaptiveAvgPool(40×40→10×10, k=4) → 10×10×512
        # out[2]: AdaptiveAvgPool(20×20→10×10, k=2) → 10×10×1024
        
        # Concatenate
        cat_out = torch.cat(out, dim=1)  # 10×10×(256+512+1024) = 10×10×1792
        
        # Conv
        return self.conv(cat_out)  # Conv(1792→352) → 10×10×352
```

#### Công Dụng:
```
Pyramid Pooling:
┌─────────────────────────────────────────────────┐
│                                                 │
│  P3(80×80×256)   P4(40×40×512)   P5(20×20×1024)│
│       │                │                │       │
│       └────────────────┼────────────────┘       │
│          All → 10×10   │                        │
│       (khác nhau pool size)                     │
│                │                                │
│            Concat: 10×10×1792                  │
│                │                                │
│            Conv: 1792→352                      │
│                │                                │
│            Output: 10×10×352                   │
└─────────────────────────────────────────────────┘

✅ Mục đích:
- Gộp 3 mức độ chi tiết
- Pool với stride=2 để giảm kích thước
- Bắt được multi-scale context
```

---

### **Layer 21: TopBasicLayer → Transformer Blocks**

#### Code:
```python
class TopBasicLayer(nn.Module):
    def __init__(self, embedding_dim, ouc_list, block_num=2, key_dim=8, num_heads=4, ...):
        super().__init__()
        self.transformer_blocks = nn.ModuleList()
        for i in range(block_num):  # 2 blocks
            self.transformer_blocks.append(top_Block(
                embedding_dim, key_dim=key_dim, num_heads=num_heads, ...
            ))
        self.conv = nn.Conv2d(embedding_dim, sum(ouc_list), 1)

    def forward(self, x):  # x: 10×10×352
        # Xử lý qua 2 Transformer blocks
        for i in range(block_num):
            x = self.transformer_blocks[i](x)
        return self.conv(x)  # 352 → sum(ouc_list)

class top_Block(nn.Module):
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.attn = Attention(dim, key_dim, num_heads)  # Multi-head attention
        self.mlp = Mlp(dim, hidden_features=dim*mlp_ratio)  # FFN

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))  # Residual + Attention
        x = x + self.drop_path(self.mlp(x))  # Residual + MLP
        return x
```

#### Công Dụng:
```
Transformer Blocks:
- 2 blocks (2× iterative refinement)
- Multi-head attention (4 heads)
- Feed-forward network (FFN)
- Residual connections (skip-add)
- DropPath (stochastic depth)

Tính toán Attention:
  Q, K, V = project(x)
  Attention = softmax(Q @ K^T / √d_k) @ V
  Multi-head: Chia Q,K,V thành 4 heads, tính riêng lẻ, concat

✅ Mục đích:
- Self-attention để bắt global dependencies
- Refine features với 2 iterations
- Output: 10×10×352 → split thành 2 phần
```

---

### **Layers 22-27: Refinement Paths**

```
Layer 22: AdvPoolFusion
  Input: [L19 (80×80×256), L16 (40×40×256)]
  
  def forward(self, x):
      x1, x2 = x
      N, C, H, W = x2.shape  # 40×40
      x1 = adaptive_avg_pool2d(x1, [H, W])  # 80×80→40×40
      return torch.cat([x1, x2], 1)  # 40×40×(256+256)=40×40×512
  
  Output: 40×40×512

Layer 23: InjectionMultiSum_Auto_pool([256, [64, 128], 0])
  Tiêm thông tin từ TopBasicLayer vào 40×40×256

Layer 24: C3Ghost(256→256)
  Output: 80×80×256

Layer 25-27: Tương tự Layer 22-24
  Refinement cho P4 (40×40×512)
```

---

## DETECTION HEAD

### **Layer 28: Detect Head → Predictions**

```python
# Final Detection
Input:
  - P3 (L19): 80×80×256    # Đối tượng nhỏ
  - P4 (L24): 40×40×256    # Đối tượng vừa (refined)
  - P5 (L27): 40×40×512    # Đối tượng lớn (refined)

Output Format (cho mỗi pyramid level):
  P3: 80×80 grid → 6400 anchors
    Mỗi anchor: [x, y, w, h, confidence, class_prob1, ..., class_prob80]
    Mỗi anchor: 4 + 1 + 80 = 85 values
    P3 total: 6400 × 85

  P4: 40×40 grid → 1600 anchors × 85
  
  P5: 20×20 grid → 400 anchors × 85

Total predictions: 8400 × 85 = 714,000 values

Decode:
  - Bounding box: (cx, cy, w, h) ← từ 4 values đầu tiên
  - Objectness: confidence ← sigmoid(5th value)
  - Class probabilities: 80 classes ← softmax(values 6-85)
  - NMS (Non-Maximum Suppression): loại bỏ boxes trùng lặp
```

---

## TÍNH TOÁN MATH

### **1. Convolution Output Size**

```
H_out = ⌊(H_in + 2×padding - dilation×(kernel-1) - 1) / stride⌋ + 1

Ví dụ: Conv(k=3, s=2, padding=1) trên 640×640
  H_out = ⌊(640 + 2×1 - 1×(3-1) - 1) / 2⌋ + 1
        = ⌊(640 + 2 - 2 - 1) / 2⌋ + 1
        = ⌊639 / 2⌋ + 1
        = 319 + 1
        = 320 ✓
```

### **2. Adaptive Average Pool**

```
AdaptiveAvgPool(input_shape, output_shape):
  Chia input thành output_shape regions
  Tính trung bình mỗi region
  
Ví dụ: 80×80 → 40×40 (chia 2×2)
  Mỗi 2×2 vùng → 1 giá trị = mean
  Output: 40×40
  
Công dụng: Downsampling với bảo toàn thông tin tốt hơn MaxPool
```

### **3. Batch Normalization**

```
y = (x - μ_batch) / √(σ_batch² + ε)  * γ + β

Trong đó:
  μ_batch = mean(x) trong batch
  σ_batch² = variance(x) trong batch
  γ, β = learnable parameters
  ε = small constant (≈1e-5)

Mục đích: 
  - Chuẩn hóa activations
  - Tốc độ training
  - Giảm internal covariate shift
```

### **4. Activation Functions**

```
ReLU:
  y = max(0, x)

Mish:
  y = x × tanh(softplus(x))
      = x × tanh(ln(1 + e^x))

h_sigmoid (Hard Sigmoid):
  y = ReLU6(x + 3) / 6
    = min(max(x + 3, 0), 6) / 6
    ≈ clip(x, 0, 1) - dùng để tính attention weights

Ưu điểm Mish:
  - Smooth, non-monotonic
  - Cải thiện gradient flow
  - Tốt hơn ReLU trong nhiều trường hợp
```

### **5. Ghost Convolution**

```
Tính toán:
  y_primary = Conv(x, kernel, channels/2)
  y_cheap = DWConv_5×5(y_primary, channels/2)
  y = cat([y_primary, y_cheap])

Độ phức tạp:
  Standard Conv(c1→c2): c1×c2×k²
  Ghost Conv(c1→c2):
    - Primary Conv: c1×(c2/2)×k²
    - Cheap DWConv: (c2/2)×5² / c2 × (c2/2) ≈ (c2/2)×5²
    - Total: ≈ 50-60% chi phí của Conv thường
```

### **6. Multi-head Attention**

```
input: B × C × H × W

Chia thành num_heads (e.g., 4):
  Q = Conv_Q(input)  → B × num_heads × key_dim × (H×W)
  K = Conv_K(input)  → B × num_heads × key_dim × (H×W)
  V = Conv_V(input)  → B × num_heads × d_value × (H×W)

Tính attention mỗi head:
  attn_i = softmax(Q_i @ K_i^T / √key_dim) @ V_i

Concat tất cả heads:
  output = Concat([attn_1, attn_2, attn_3, attn_4])
  output = Conv_out(output)

Mục đích:
  - Bắt được dependencies từ nhiều "góc nhìn" khác nhau
  - Chuyển từ local spatial convolution → global attention
```

---

## 📊 TÓMALAYERS TOPOLOGY

```
640×640×3
    ↓ [0] Conv 3×3 s=2
320×320×64
    ↓ [1] Conv 3×3 s=2
160×160×128
    ↓ [2] C2f ✓ P2
160×160×128
    ↓ [3] Conv 3×3 s=2
80×80×256
    ↓ [4] C2f ✓ P3
80×80×256
    ↓ [5] Conv 3×3 s=2
40×40×512
    ↓ [6] C2f ✓ P4
40×40×512
    ↓ [7] Conv 3×3 s=2
20×20×1024
    ↓ [8] C2f
20×20×1024
    ↓ [9] SimSPPF ✓ P5
20×20×1024

════════════════════ HEAD ════════════════════

[10] SimFusion_4in([P2, P3, P4, P5]) → 40×40×1920
    ↓
[11] IFM → 40×40×96 → [64, 32]

     BRANCH 1 (P4 Path):
[12] Conv(P5) → 20×20×512
[13] SimFusion_3in([P3, P4, Conv(L12)]) → 40×40×512
[14] InjectionMultiSum([40×40×512], [64,32], flag=0) → 40×40×512
[15] C3Ghost → 40×40×512

     BRANCH 2 (P3 Path):
[16] Conv(P4) → 40×40×256
[17] SimFusion_3in([P2, P3, Conv(L16)]) → 40×40×256
[18] InjectionMultiSum([40×40×256], [64,32], flag=1) → 40×40×256
[19] C3Ghost → 80×80×256 ✓ P3_refined

[20] PyramidPoolAgg([P3_refined, P4, P5]) → 10×10×352
[21] TopBasicLayer(transformer) → 10×10×352

     REFINEMENT:
[22] AdvPoolFusion([P3_refined, Conv(L16)]) → 40×40×512
[23] InjectionMultiSum → 40×40×256
[24] C3Ghost → 80×80×256 ✓ P3_final

[25] AdvPoolFusion([P3_final, Conv(L12)]) → 40×40×512
[26] InjectionMultiSum → 40×40×512
[27] C2f → 40×40×512 ✓ P5_final

[28] Detect([P3_final, P4, P5_final]) → 80×80 + 40×40 + 20×20
     → 6400 + 1600 + 400 = 8400 predictions
     → mỗi: [x, y, w, h, conf, 80 class probs]
```

---

## 🎯 Tóm Tắt

| Layer | Type | Input Size | Output Size | Purpose |
|-------|------|-----------|-----------|---------|
| 0-9 | Backbone | 640×640×3 | 20×20×1024 (P5) | Feature extraction |
| 10-11 | Init fusion | [P2,P3,P4,P5] | 40×40×96 (IFM) | Multi-scale combination |
| 12-15 | P4 branch | [P3,P4,P5] | 40×40×512 | Medium object detection |
| 16-19 | P3 branch | [P2,P3,P4] | 80×80×256 | Small object detection |
| 20-21 | Aggregation | [P3,P4,P5] | 10×10×352 | Pyramid context + Transformer |
| 22-27 | Refinement | Pyramid | P3,P4,P5 refined | Enhance predictions |
| 28 | Detect | 3 scales | 8400 predictions | Final detections |

