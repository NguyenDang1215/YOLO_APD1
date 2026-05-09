# 🗜️ Công Nghệ Nén Mô Hình YOLO8 GoldACsim

## 📊 Bảng Tổng Hợp Các Phương Pháp

| Phương Pháp | Mức Nén | Mất Accuracy | Độ Khó | Chi Phí | Công nghệ |
|------------|---------|-------------|--------|---------|-----------|
| **Quantization (INT8)** | 4× | 0.5-2% | ⭐ | Thấp | TVM, TensorRT, CoreML |
| **Pruning Structured** | 3-5× | 1-3% | ⭐⭐ | Thấp | TorchPruning |
| **Knowledge Distillation** | 2-3× | 0.5-1.5% | ⭐⭐⭐ | Trung bình | PyTorch, TensorFlow |
| **Low-Rank Approximation** | 2-4× | 1-2% | ⭐⭐⭐ | Trung bình | TensorDecomp |
| **Mixed Precision** | 2× | <0.5% | ⭐ | Thấp | TensorRT, ONNX Runtime |
| **Neural Architecture Search** | 3-8× | <1% | ⭐⭐⭐⭐ | Cao | AutoML, DARTS |
| **Lottery Ticket Hypothesis** | 5-10× | <2% | ⭐⭐⭐ | Cao | PyTorch |
| **Knowledge Distillation + Quantization** | 8-12× | 1-2% | ⭐⭐⭐⭐ | Cao | Hybrid |

---

## 1️⃣ QUANTIZATION (Lượng Tử Hóa)

### 1.1 INT8 Quantization

#### Công Thức

```
Quantization:
  q = round(x / scale) + zero_point
  
Trong đó:
  x = floating point value
  scale = (x_max - x_min) / (255 - 0) = range / 255
  zero_point = round(-x_min / scale)
  q = quantized value (8-bit int)

De-quantization:
  x_approx = (q - zero_point) × scale
```

#### Ví Dụ Tính Toán

```python
import numpy as np

# Input: floating point tensor
x = np.array([[-2.5, -1.2, 0.8, 1.5, 3.2]])

# Tìm range
x_min, x_max = x.min(), x.max()  # -2.5, 3.2
range_val = x_max - x_min  # 5.7

# Tính scale
scale = range_val / 255  # 5.7 / 255 ≈ 0.0224
zero_point = round(-x_min / scale)  # round(2.5 / 0.0224) ≈ 112

# Quantize
q = np.round(x / scale + zero_point)  # INT8: [0, 53, 147, 169, 255]
q = np.clip(q, 0, 255)  # Clip to INT8 range

# De-quantize
x_approx = (q - zero_point) * scale
# ≈ [[-2.51, -1.17, 0.81, 1.56, 3.19]]

Lỗi:
  Error = x - x_approx
  ≈ [0.01, 0.03, -0.01, -0.06, 0.01]
  Max Error ≈ 0.06 (rất nhỏ!)
```

#### Các Loại Quantization

```
1. Post-Training Quantization (PTQ):
   - Chỉ cần đổ data qua mô hình 1 lần
   - Tính scale/zero_point từ activation distribution
   - Nhanh, đơn giản, mất 1-2% accuracy
   
   Formula:
     Dữ liệu calibration: x_1, x_2, ..., x_N
     scale = (max(x) - min(x)) / 255
     zero_point = argmin |x_quantized - x_original|

2. Quantization-Aware Training (QAT):
   - Training trong suốt quá trình quantization
   - Mô hình học cách thích ứng với INT8
   - Chậm nhưng chính xác hơn (mất <0.5% accuracy)
   
   Formula:
     forward: x_q = quantize(x)
     loss: L = MSE(model(x_q), y) + λ × TV(q)
     TV = Total Variation penalty
     
   Backprop qua fake quantization:
     ∂L/∂x = ∂L/∂x_q × ∂x_q/∂x

3. Dynamic Quantization:
   - Tính scale trong runtime (mỗi batch khác nhau)
   - Flexible nhưng chậm hơn
   
4. Symmetric vs Asymmetric:
   - Symmetric: [-127, 127] (đối xứng)
   - Asymmetric: [0, 255] (không đối xứng, tốt hơn)
```

#### Code Thực Tế (PyTorch)

```python
import torch
import torch.quantization as tq

# Model chuẩn bị
model = YOLOv8GoldACsim(...)
model.eval()

# Post-Training Quantization
model_quantized = tq.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},  # Layers to quantize
    dtype=torch.qint8
)

# Lưu model
torch.save(model_quantized, 'model_int8.pt')

# So sánh kích thước
original_size = sum(p.numel() * 4 for p in model.parameters()) / 1024**2  # MB
quantized_size = sum(p.numel() * 1 for p in model_quantized.parameters()) / 1024**2  # 4× nhỏ hơn
print(f"Original: {original_size:.2f}MB → Quantized: {quantized_size:.2f}MB")

# Accuracy check
with torch.no_grad():
    acc_original = evaluate(model, val_loader)
    acc_quantized = evaluate(model_quantized, val_loader)
    print(f"Accuracy drop: {(acc_original - acc_quantized) * 100:.2f}%")
```

#### Công Thức Tính Accuracy Drop

```
Nếu muốn đảm bảo accuracy drop < 1%:

Heuristic formulas:
  1. Per-channel vs Per-tensor:
     Per-channel accuracy: ~5% tốt hơn per-tensor
     
  2. Activation vs Weight quantization:
     Quantize weights: ~2-3% drop
     Quantize activation: ~1-2% drop
     Cả hai: ~3-5% drop
     
  3. Warmup formula:
     Để QAT thành công:
     
     warmup_epoch = total_epoch / 5
     
     loss = MSE_loss + β × quantization_error
     
     β_schedule = β_0 × (1 - t/T) ^ 2  # Cosine annealing
     
     Nếu β quá lớn: underfitting
     Nếu β quá nhỏ: không converge
     
  4. Batch size effect:
     Batch size nhỏ → Per-batch statistics → Lớn variation
     Batch size lớn → Ổn định → Kết quả tốt hơn
     
     Recommend batch_size = 32-64 cho QAT
```

#### Calibration Dataset

```
Calibration để tìm scale/zero_point tối ưu:

Method 1: KL Divergence (Kullback-Leibler)
  Minimize KL(P_fp32 || P_int8)
  P = distribution của activations
  
  Formula:
    KL(P||Q) = Σ P(x) × log(P(x)/Q(x))
    
  Cách làm:
    1. Collect activation từ 100-200 calibration samples
    2. Tính histogram
    3. Thử 255 scales khác nhau
    4. Chọn scale → min KL divergence

Method 2: Entropy-based
  Find scale → maximize entropy (information preservation)
  
Method 3: Min-Max (đơn giản):
  scale = (max(x) - min(x)) / 255
  Dễ nhưng kém hiệu quả (outliers ảnh hưởng)

Recommendation:
  - Use KL-divergence cho tốt nhất
  - Cần 100-500 calibration samples
  - Samples nên đại diện cho data distribution
```

---

## 2️⃣ PRUNING (Cắt Tỉa)

### 2.1 Structured Pruning (Cắt Toàn Bộ Channels)

#### Công Thức

```
L1/L2 Norm-based Pruning:

Cho mỗi filter/channel:
  L1_norm(w) = Σ |w_i|
  L2_norm(w) = √(Σ w_i²)

Xếp hạng channels theo magnitude:
  ranks = sort(L1_norms)
  
Cắt tỉa những channels có norm nhỏ nhất:
  pruning_ratio = số channels bị xóa / tổng channels
  
Ví dụ:
  Layer có 512 channels
  pruning_ratio = 0.3 (30%)
  → Xóa 154 channels có L1_norm nhỏ nhất
  → Còn lại 358 channels
  
Công dụng:
  - Tương đương với bỏ các connections yếu
  - Mô hình học bỏ qua 30% channels không quan trọng
```

#### Công Thức Tổn Thất Accuracy

```
Empirical formula (từ các papers):

Δ_accuracy ≈ pruning_ratio × scaling_factor

Scaling factor phụ thuộc vào:
  - Layer importance:
    Early layers: more important (scale_factor = 10)
    Middle layers: (scale_factor = 5)
    Last layers: less important (scale_factor = 2)
    
  - Task complexity:
    Detection: 0.05-0.10 accuracy drop per 10% pruning
    Classification: 0.02-0.05 drop per 10% pruning
    
  - Model capacity:
    Over-parameterized: can prune 50% safely
    Optimized: pruning 20% giới hạn an toàn

Ví dụ:
  Pruning 30% channels từ middle layer:
  Δ_accuracy ≈ 0.30 × 0.07 ≈ 2.1% drop
  
  Nếu muốn drop < 1%:
  pruning_ratio < 0.14 (14% channels)
```

#### Code Thực Tế (PyTorch)

```python
import torch
import torch.nn.utils.prune as prune

model = YOLOv8GoldACsim(...)

# Method 1: L1 Unstructured Pruning
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.3)  # Prune 30%
        # Sau đó phải "remove" để thực sự giảm memory
        prune.remove(module, 'weight')

# Method 2: Structured Pruning (cắt channels)
class ChannelPruner:
    def __init__(self, model, pruning_ratio=0.3):
        self.model = model
        self.ratio = pruning_ratio
    
    def compute_channel_importance(self, module):
        # L2-norm của weights
        if hasattr(module, 'weight'):
            w = module.weight.data
            # Shape: [out_channels, in_channels, k, k]
            norms = torch.norm(w.view(w.size(0), -1), p=2, dim=1)
            return norms
        return None
    
    def prune(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                norms = self.compute_channel_importance(module)
                if norms is not None:
                    # Tính ngưỡng pruning
                    threshold = torch.quantile(norms, self.ratio)
                    # Mask: giữ channels có norm > threshold
                    mask = norms > threshold
                    
                    # Áp dụng mask (trừu tượng)
                    # Thực tế cần thay đổi architecture
                    print(f"Prune {(~mask).sum()} channels từ {len(mask)}")

pruner = ChannelPruner(model, pruning_ratio=0.3)
pruner.prune()

# Fine-tune sau pruning
train_with_pruned_model(model, epochs=10)
```

#### Tính Toán Compression Ratio

```
Compression Ratio = Original_FLOPs / Pruned_FLOPs

FLOP = Floating Point Operations

Công thức FLOP cho Conv layer:
  FLOP = H × W × Cin × Kh × Kw × Cout
  
Ví dụ Layer 15 (C3Ghost):
  Input: 40×40×512
  Output: 40×40×512
  Kernel: 3×3
  
  Original FLOP:
    = 40 × 40 × 512 × 3 × 3 × 512
    = 1,572,864,000 FLOPs ≈ 1.57 GFLOPs
  
  Sau khi cắt 30% channels:
    Cout = 512 × 0.7 = 358 channels
    New FLOP:
    = 40 × 40 × 512 × 3 × 3 × 358
    ≈ 1.1 GFLOPs
  
  Compression ratio ≈ 1.57 / 1.1 ≈ 1.43×
```

---

## 3️⃣ KNOWLEDGE DISTILLATION (Chuyển Giao Kiến Thức)

### 3.1 Công Thức

```
Knowledge Distillation:
  
Teacher Network (đã trained, lớn):
  - YOLOv8 Large (lớn hơn)
  - Accuracy cao nhưng chậm

Student Network (nhỏ):
  - YOLOv8 GoldACsim (compact)
  - Muốn học từ teacher

Loss Function:
  L_total = α × L_ce + (1-α) × L_kd
  
Trong đó:
  L_ce = Cross-entropy loss (ground truth)
  L_kd = KL divergence (student vs teacher)
  α = 0.1-0.3 (balance factor)
  
Knowledge Transfer:
  p_teacher = softmax(z_teacher / T)  # Soft target
  p_student = softmax(z_student / T)  # Soft prediction
  
  L_kd = T² × KL(p_teacher || p_student)
       = T² × Σ p_teacher × log(p_teacher / p_student)

Temperature T:
  T > 1: Làm mềm probability distribution
  T = 1: Normal softmax
  T = 3-10: Phù hợp với detection
  T cao → Thông tin fine-grained nhiều hơn
```

#### Ví Dụ Tính Toán

```
Teacher outputs bounding box: [x, y, w, h, conf]
  Logits: [0.5, 1.2, -0.3, 0.8, 2.1]
  
Student outputs:
  Logits: [0.3, 0.9, -0.5, 0.6, 1.5]

Temperature T = 5:
  Teacher soft probs:
    p_t = softmax([0.5, 1.2, -0.3, 0.8, 2.1] / 5)
        = softmax([0.1, 0.24, -0.06, 0.16, 0.42])
        ≈ [0.19, 0.20, 0.18, 0.19, 0.24]
        
  Student soft probs:
    p_s = softmax([0.3, 0.9, -0.5, 0.6, 1.5] / 5)
        ≈ [0.19, 0.20, 0.17, 0.19, 0.25]
  
  KL divergence:
    KL = Σ p_t × log(p_t / p_s)
        ≈ 0.001  (rất nhỏ, có thể học)
        
  Nếu không có T:
    p_t (hard) = [0, 0, 0, 0, 1]  # One-hot (quá khó)
    p_s (hard) = [0, 0, 0, 0, 1]
    KL = 0 (mất signal học)
```

#### Code Thực Tế

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=5.0, alpha=0.1):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, targets):
        # Cross-entropy loss (học từ ground truth)
        ce = self.ce_loss(student_logits, targets)
        
        # Knowledge distillation loss
        # Soft targets từ teacher
        teacher_soft = F.softmax(teacher_logits / self.T, dim=1)
        student_soft = F.log_softmax(student_logits / self.T, dim=1)
        
        kd = self.kl_loss(student_soft, teacher_soft)
        
        # Combine
        loss = self.alpha * ce + (1 - self.alpha) * (self.T ** 2) * kd
        return loss

# Training loop
teacher_model = YOLOv8Large(pretrained=True)
teacher_model.eval()  # Freeze teacher

student_model = YOLOv8GoldACsim()
student_optimizer = torch.optim.AdamW(student_model.parameters(), lr=0.001)

distill_loss_fn = DistillationLoss(temperature=5.0, alpha=0.1)

for epoch in range(100):
    for batch_idx, (images, targets) in enumerate(train_loader):
        student_optimizer.zero_grad()
        
        # Forward
        with torch.no_grad():
            teacher_logits = teacher_model(images)
        student_logits = student_model(images)
        
        # Loss
        loss = distill_loss_fn(student_logits, teacher_logits, targets)
        
        # Backward
        loss.backward()
        student_optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}")
```

#### Tính Toán Accuracy Gain

```
Empirical Formula (từ papers):

Δ_accuracy_distilled ≈ 0.7-0.9 × Δ_accuracy_baseline

Ví dụ:
  Teacher accuracy: 95%
  Student baseline: 92% (3% gap)
  
  Với distillation:
  Expected student accuracy:
    = 95% - 0.3 × 3%
    ≈ 94.1%
  
  Gain = 94.1% - 92% = 2.1% improvement!

Factors ảnh hưởng:
  - Teacher quality: Càng tốt, student càng học tốt
  - Temperature: 
    T=1: Hard targets, ít information
    T=5-10: Mềm targets, nhiều information
    T>20: Quá mềm, mất specificity
    
  - Alpha balance:
    α=0.1: Chủ yếu theo teacher (distillation-heavy)
    α=0.5: Cân bằng
    α=0.9: Chủ yếu theo ground truth
    
  Best practice: α=0.1-0.3, T=5-10
```

---

## 4️⃣ LOW-RANK APPROXIMATION

### 4.1 Công Thức

```
SVD (Singular Value Decomposition):

W = U × Σ × V^T

Trong đó:
  W: Original weight matrix (d1 × d2)
  U: (d1 × r)
  Σ: Diagonal matrix (r × r) - singular values
  V: (d2 × r)
  r = min(d1, d2)

Low-rank approximation (rank k < r):
  W_approx = U_k × Σ_k × V_k^T
  
  Chỉ lấy k singular values lớn nhất
  Bỏ đi những small singular values (noise)

Compression:
  Original storage: d1 × d2
  Low-rank storage: k × (d1 + d2)
  Compression ratio: (d1 × d2) / (k × (d1 + d2))

Ví dụ Layer 15 (C3Ghost):
  Conv layer: 512 × 512 × 3 × 3 = 2.36M parameters
  
  Reshape to 2D: (512×3×3) × 512 = 4608 × 512
  
  Low-rank k=256:
  Compressed: 256 × (4608 + 512) = 1.31M
  Compression ratio: 2.36M / 1.31M ≈ 1.8×
```

#### Code Thực Tế

```python
import torch
import numpy as np
from scipy.linalg import svd

def low_rank_decompose(weight_matrix, rank=None):
    """
    Decompose weight matrix to low-rank approximation
    
    Args:
        weight_matrix: (d1, d2) weight matrix
        rank: target rank (if None, keep 95% information)
    
    Returns:
        U, Sigma, V: Low-rank factors
    """
    # Compute SVD
    U, S, Vt = svd(weight_matrix, full_matrices=False)
    
    if rank is None:
        # Keep 95% energy
        total_energy = np.sum(S ** 2)
        energy_threshold = 0.95 * total_energy
        
        cumsum_energy = 0
        for i, s in enumerate(S):
            cumsum_energy += s ** 2
            if cumsum_energy >= energy_threshold:
                rank = i + 1
                break
    
    # Keep top-k singular values
    U_k = U[:, :rank]
    S_k = S[:rank]
    V_k = Vt[:rank, :]
    
    return U_k, S_k, V_k

def apply_low_rank_to_model(model, target_rank=256):
    """Apply low-rank decomposition to all Conv layers"""
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # Get weight
            w = module.weight.data.cpu().numpy()  # (out_c, in_c, k, k)
            
            # Reshape to 2D
            out_c, in_c, kh, kw = w.shape
            w_2d = w.reshape(out_c, -1)  # (out_c, in_c×k×k)
            
            # SVD
            U, S, V = low_rank_decompose(w_2d, rank=target_rank)
            
            # Reconstruct
            w_approx = U @ np.diag(S) @ V
            w_approx = w_approx.reshape(out_c, in_c, kh, kw)
            
            # Update module
            module.weight.data = torch.from_numpy(w_approx).float()
            
            print(f"{name}: {w.size} → {U.size} + {V.size} " +
                  f"(compression: {w.size / (U.size + V.size):.2f}×)")

model = YOLOv8GoldACsim(...)
apply_low_rank_to_model(model, target_rank=256)

# Fine-tune
train_with_low_rank_model(model, epochs=20)
```

#### Tính Toán Information Retention

```
Energy-based formula:

Information retained = (Σ_{i=1}^k S_i²) / (Σ_{i=1}^r S_i²)

Ví dụ:
  Singular values: [100, 50, 30, 15, 5, 2, 1]
  Total energy: 100² + 50² + 30² + ... = 14739
  
  Keep k=3: Energy = 100² + 50² + 30² = 12400
  Retained: 12400/14739 ≈ 84%
  
  Keep k=4: Energy = 12400 + 15² = 12625
  Retained: 12625/14739 ≈ 85.7%

Accuracy drop formula:
  Δ_accuracy ≈ (1 - information_retained) × scaling_factor
  
  Ví dụ:
    Information: 85% → Loss = 15%
    Δ_accuracy ≈ 0.15 × 0.1 ≈ 1.5% drop
    
  Để drop < 1%:
    Need information > 90%
```

---

## 5️⃣ MIXED PRECISION (FP32 + FP16)

### 5.1 Công Thức

```
FP32 (float32): 32-bit
  1 sign bit + 8 exponent bits + 23 mantissa bits
  Range: ~10^-38 to 10^38
  Precision: ~7 decimal digits

FP16 (float16): 16-bit
  1 sign bit + 5 exponent bits + 10 mantissa bits
  Range: ~10^-4 to 10^4
  Precision: ~3-4 decimal digits

Mixed Precision Strategy:
  - Forward pass:
    * Weights: FP32 (chính xác)
    * Activations: FP16 (nhanh)
    * Loss scale factor: s (để tránh underflow)
    
  - Backward pass:
    * Gradient: FP16 (nhanh)
    * Accumulate: FP32 (chính xác)

Loss scaling formula:
  loss_scaled = loss × s
  
  Trong đó:
    s = 2^15 (thường dùng)
    Tránh gradient underflow trong FP16
    
  Backward:
    grad_scaled = ∂loss_scaled/∂w = s × ∂loss/∂w
    grad_unscaled = grad_scaled / s = ∂loss/∂w
```

#### Code Thực Tế (PyTorch + Automatic Mixed Precision)

```python
from torch.cuda.amp import autocast, GradScaler

model = YOLOv8GoldACsim(...).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# GradScaler for automatic loss scaling
scaler = GradScaler()

for epoch in range(100):
    for images, targets in train_loader:
        optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast(dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        # Scaled backward
        scaler.scale(loss).backward()
        
        # Unscale gradients
        scaler.unscale_(optimizer)
        
        # Gradient clipping (optional)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

# Inference
model.half()  # Convert to FP16
with torch.no_grad():
    with autocast(dtype=torch.float16):
        predictions = model(test_images)

# Memory savings
print(f"Model memory: {sum(p.numel() * 2 for p in model.parameters()) / 1024**2:.2f}MB (halved!)")
```

#### Tính Toán Memory & Speed

```
Memory savings:
  FP32: 4 bytes per value
  FP16: 2 bytes per value
  Savings: 2× memory reduction
  
Speed improvements:
  GPU tensor cores (Volta, Ampere):
    FP32: 125 TFLOPS
    FP16: 250+ TFLOPS (2-4× faster)
  
  Modern GPU (A100):
    FP32: 312 TFLOPS
    TF32: 625 TFLOPS
    FP16: 625 TFLOPS
    INT8: 1250 TFLOPS

Accuracy Impact:
  Most detection tasks: < 0.5% drop
  Nhạy cảm layers (loss calculation):
    Dùng FP32 riêng biệt
    
  Formula:
    Δ_accuracy_fp16 ≈ 0.1-0.3%
```

---

## 6️⃣ LOTTERY TICKET HYPOTHESIS

### 6.1 Công Thức

> "A randomly initialized, dense neural network contains a subnetwork that is initialized such that - **when trained in isolation** - it can match the test accuracy of the original network after training for the same number of iterations."

#### Thuật Toán

```
1. Random initialization: W_0 ~ N(0, σ²)

2. Train k iterations: W_k = Train(W_0, data, k)

3. Prune by magnitude:
   Keep top-p% weights where |W_k| is largest
   W_k_pruned = W_k ⊙ mask  (⊙ = element-wise multiply)
   
4. Reset to initialization:
   W_k_reset = W_0 ⊙ mask
   
5. Train again k iterations:
   W_k' = Train(W_k_reset, data, k)
   
Result: W_k' có accuracy ~ W_k (hoặc tốt hơn!)

Key insight:
  - Không phải là weights mà là connectivity pattern quan trọng
  - Đúng masks từ đầu sẽ giúp training converge tốt
  - Có thể tìm masks "may mắn" để training vẫn tốt
```

#### Công Thức Độ Chính Xác

```
Winning ticket accuracy (WT):
  Acc_WT ≥ Acc_dense - ε
  
Trong đó:
  ε = 0-2% (nhỏ)
  Acc_dense = accuracy của mô hình ban đầu
  
Sparsity (tỷ lệ prune):
  Sparsity = (1 - #kept_weights / #total_weights) × 100%
  
Compression ratio:
  CR = #total_weights / #kept_weights
  
Ví dụ:
  Original model: 25M parameters, 95% accuracy
  
  Find winning ticket:
    - Prune 90% weights (keep 10%)
    - Reset & retrain
    - Đạt được 94-95% accuracy lại!
    
  Sparsity: 90%
  Compression: 10×
```

#### Code Thực Tế

```python
import torch
import torch.nn.utils.prune as prune
from copy import deepcopy

def find_winning_ticket(model, train_loader, num_epochs=100, prune_ratio=0.9):
    """Find lottery ticket through iterative pruning"""
    
    # 1. Store initial weights
    initial_state = deepcopy(model.state_dict())
    
    # 2. Train model
    train_model(model, train_loader, epochs=num_epochs)
    trained_weights = deepcopy(model.state_dict())
    
    # 3. Prune by magnitude
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # Magnitude pruning
            prune.l1_unstructured(module, name='weight', amount=prune_ratio)
            prune.remove(module, 'weight')  # Make permanent
    
    # 4. Reset weights to initialization
    pruned_structure = deepcopy(model.state_dict())
    
    for key in pruned_structure:
        if 'weight' in key:
            # Get mask (non-zero elements)
            mask = pruned_structure[key] != 0
            # Reset to initialization
            model.state_dict()[key].copy_(initial_state[key] * mask.float())
    
    # 5. Retrain from reset weights
    accuracy_after = train_model(model, train_loader, epochs=num_epochs)
    
    return model, accuracy_after

# Usage
model = YOLOv8GoldACsim(...)
winning_model, acc = find_winning_ticket(model, train_loader, 
                                          num_epochs=100, 
                                          prune_ratio=0.9)

print(f"Winning ticket achieved {acc:.2f}% accuracy with 90% sparsity!")
```

---

## 7️⃣ HYBRID: QUANTIZATION + DISTILLATION + PRUNING

### 7.1 Kết Hợp Các Phương Pháp

```
Pipeline tối ưu:

Step 1: Knowledge Distillation (3-4 epochs)
  Teacher → Student (giảm 30% model size)
  Accuracy: 95% → 94.5% (drop: 0.5%)

Step 2: Pruning (20 epochs)
  Cắt 40% channels từ student
  Accuracy: 94.5% → 93.8% (drop: 0.7%)

Step 3: Quantization-Aware Training (20 epochs)
  INT8 quantization với QAT
  Accuracy: 93.8% → 93.2% (drop: 0.6%)

Total compression: 3× (distill) × 1.67× (prune) × 4× (quantization)
                 ≈ 20× compression!

Final accuracy: 93.2% (từ 95% gốc)
Accuracy drop: 1.8% (chấp nhận được)
```

#### Code Thực Tế (Hybrid Pipeline)

```python
import torch
from torch import nn, optim

class HybridCompressionPipeline:
    def __init__(self, teacher_model, student_model, train_loader, val_loader):
        self.teacher = teacher_model
        self.student = student_model
        self.train_loader = train_loader
        self.val_loader = val_loader
    
    def step1_distillation(self, epochs=5):
        """Phase 1: Knowledge Distillation"""
        print("=" * 50)
        print("Phase 1: Knowledge Distillation")
        print("=" * 50)
        
        from torch.nn.functional import kl_div, log_softmax, softmax
        
        optimizer = optim.AdamW(self.student.parameters(), lr=0.001)
        temp = 5.0
        alpha = 0.1
        
        for epoch in range(epochs):
            total_loss = 0
            for images, targets in self.train_loader:
                optimizer.zero_grad()
                
                with torch.no_grad():
                    teacher_logits = self.teacher(images)
                student_logits = self.student(images)
                
                # Distillation loss
                ce_loss = nn.CrossEntropyLoss()(student_logits, targets)
                kd_loss = kl_div(
                    log_softmax(student_logits / temp, dim=1),
                    softmax(teacher_logits / temp, dim=1)
                )
                
                loss = alpha * ce_loss + (1 - alpha) * (temp ** 2) * kd_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            val_acc = self.evaluate()
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Val Acc: {val_acc:.2f}%")
    
    def step2_pruning(self, prune_ratio=0.4, epochs=20):
        """Phase 2: Structured Pruning"""
        print("=" * 50)
        print(f"Phase 2: Pruning {prune_ratio*100:.0f}% channels")
        print("=" * 50)
        
        import torch.nn.utils.prune as prune
        
        # Apply pruning
        for module in self.student.modules():
            if isinstance(module, nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=prune_ratio)
        
        # Fine-tune
        optimizer = optim.AdamW(self.student.parameters(), lr=0.0005)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for images, targets in self.train_loader:
                optimizer.zero_grad()
                logits = self.student(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            val_acc = self.evaluate()
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Val Acc: {val_acc:.2f}%")
    
    def step3_quantization(self, epochs=20):
        """Phase 3: Quantization-Aware Training"""
        print("=" * 50)
        print("Phase 3: Quantization-Aware Training (INT8)")
        print("=" * 50)
        
        from torch.quantization import convert, prepare_qat
        from torch.quantization.quantize_fx import prepare_fx, convert_fx
        
        # Prepare for QAT
        self.student.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        self.student_qat = prepare_qat(self.student)
        
        optimizer = optim.AdamW(self.student_qat.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for images, targets in self.train_loader:
                optimizer.zero_grad()
                logits = self.student_qat(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            val_acc = self.evaluate()
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Val Acc: {val_acc:.2f}%")
        
        # Convert to quantized model
        self.student_quantized = convert(self.student_qat)
    
    def evaluate(self):
        """Evaluate accuracy"""
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in self.val_loader:
                logits = self.student(images)
                _, predicted = torch.max(logits, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        return 100 * correct / total
    
    def run_full_pipeline(self):
        """Run complete compression pipeline"""
        print("\n" + "=" * 50)
        print("HYBRID COMPRESSION PIPELINE")
        print("=" * 50 + "\n")
        
        baseline_acc = self.evaluate()
        print(f"Baseline accuracy: {baseline_acc:.2f}%\n")
        
        self.step1_distillation(epochs=5)
        print()
        self.step2_pruning(prune_ratio=0.4, epochs=20)
        print()
        self.step3_quantization(epochs=20)
        
        final_acc = self.evaluate()
        print(f"\nFinal accuracy: {final_acc:.2f}%")
        print(f"Accuracy drop: {baseline_acc - final_acc:.2f}%")
        print("=" * 50)

# Usage
teacher = YOLOv8Large(pretrained=True)
student = YOLOv8GoldACsim()

pipeline = HybridCompressionPipeline(teacher, student, train_loader, val_loader)
pipeline.run_full_pipeline()
```

---

## 📚 BẢNG SO SÁNH PAPERS & CÔNG THỨC

| Paper | Năm | Phương Pháp | Compression | Accuracy Drop |
|-------|-----|-----------|-------------|---------------|
| Deep Compression | 2016 | Pruning + Quantization + Huffman | 35-49× | <1% |
| MobileNet | 2017 | Depthwise Separable Conv | 8-9× | <2% |
| Knowledge Distillation | 2015 | Soft targets | 3× | 0.5-1.5% |
| Mixed Precision | 2017 | FP32 + FP16 | 2× | <0.5% |
| Lottery Ticket | 2019 | Magnitude pruning + reset | 10-100× | <2% |
| Quantization-Aware Training | 2018 | Simulated INT8 during training | 4× | <1% |
| YOLO4-tiny | 2020 | Architecture redesign | 188× | ~5% loss vs full |
| EdgeAI | 2021 | Hybrid approach | 20-50× | 1-3% |

---

## 🎯 RECOMMENDATION FOR YOLO8 GoldACsim

### Best Strategy (Balanced)

```
For maximum compression with <2% accuracy drop:

1. Knowledge Distillation (teacher = YOLOv8Large)
   - Temperature: 5-7
   - Alpha: 0.2
   - Epochs: 50
   → Expected compression: 0.8× (actually refined, not compressed)
   → Accuracy gain: +0.5-1%

2. Structured Pruning (40% channels)
   - Method: L2-norm based
   - Fine-tune: 30 epochs
   → Compression: 1.67×
   → Accuracy drop: 0.8-1.2%

3. Quantization-Aware Training (INT8)
   - Per-channel quantization
   - Epochs: 30
   → Compression: 4×
   → Accuracy drop: 0.5-0.8%

Total: 1.67 × 4 ≈ 6.7× compression
Total accuracy drop: 0.5-1.5% (acceptable)

For 25.9M → 3.9M parameters (~6.7× smaller)
```

### Alternative: Maximum Compression

```
For maximum compression (accepting 3-4% drop):

1. Aggressive Pruning (60% channels)
   → 2.5× compression
   → ~2% accuracy drop
   → Fine-tune: 50 epochs

2. Quantization + Mixed Precision
   → 4× compression (FP16)
   → ~0.5% additional drop
   
3. Low-rank Approximation (rank=128)
   → 1.8× compression
   → ~1% accuracy drop

Total: 2.5 × 4 × 1.8 ≈ 18× compression
Accuracy drop: 3-4%
Final model: 25.9M → 1.4M (very tiny!)
```

---

## 🔗 CÔNG THỨC KIỂM ĐỊNH

### Kiểm Tra Accuracy Drop

```python
def validate_compression_quality(original_model, compressed_model, val_loader):
    """
    Validate compressed model quality
    
    Công thức:
      - Accuracy drop: Δ = Acc_orig - Acc_comp
      - Compression ratio: CR = size_orig / size_comp
      - Quality metric: Q = CR / (1 + k × Δ)
        (k = sensitivity factor, thường 1-10)
    
    Acceptable ranges:
      - Δ < 1%: Excellent (almost imperceptible)
      - 1% < Δ < 2%: Good (acceptable)
      - 2% < Δ < 3%: Fair (noticeable)
      - Δ > 3%: Poor (too much loss)
    """
    
    original_acc = evaluate(original_model, val_loader)
    compressed_acc = evaluate(compressed_model, val_loader)
    
    drop = original_acc - compressed_acc
    
    # Calculate model sizes
    orig_size = sum(p.numel() * 4 for p in original_model.parameters()) / 1024**2
    comp_size = sum(p.numel() * 2 for p in compressed_model.parameters()) / 1024**2
    
    compression_ratio = orig_size / comp_size
    
    # Quality metric
    k = 5  # sensitivity factor
    quality_score = compression_ratio / (1 + k * drop)
    
    print(f"Original accuracy: {original_acc:.2f}%")
    print(f"Compressed accuracy: {compressed_acc:.2f}%")
    print(f"Accuracy drop: {drop:.2f}%")
    print(f"Compression ratio: {compression_ratio:.2f}×")
    print(f"Quality score: {quality_score:.2f}")
    
    # Verdict
    if drop < 0.5:
        print("✅ Excellent - imperceptible loss")
    elif drop < 1.0:
        print("✅ Good - acceptable")
    elif drop < 2.0:
        print("⚠️ Fair - noticeable but tolerable")
    elif drop < 3.0:
        print("❌ Poor - significant loss")
    else:
        print("❌ Unacceptable - too much loss")
    
    return {
        'accuracy_drop': drop,
        'compression_ratio': compression_ratio,
        'quality_score': quality_score
    }
```

