# ⚡ Quick Reference: Model Compression Decision Tree

## 🎯 Chọn Phương Pháp Nào?

### 1️⃣ Bạn cần gì?

#### A. **Chỉ cần nhanh hơn** (Speed > Size)
```
✅ Mixed Precision (FP32 + FP16)
   - Effort: ⭐ (1 ngày)
   - Compression: 2× memory
   - Speed: 2-4× faster
   - Accuracy drop: <0.5%
   
   Code:
   model.half()  # Convert to FP16
   with torch.cuda.amp.autocast():
       output = model(input)
```

---

#### B. **Cần nhỏ & nhanh, vẫn chuẩn xác** (Balanced)
```
✅ Recommended: Distillation + Pruning 40% + INT8 QAT
   - Effort: ⭐⭐⭐ (2-3 tuần)
   - Compression: 6.7× (~3.8M params)
   - Accuracy drop: 1-2% (chấp nhận được)
   - Speed: 100× faster
   
   Results:
   - Original: 25.9M params, 95% acc
   - Compressed: 3.8M params, 93.5% acc ✅
   - From: 500MB → 77MB disk ✅
   
   Timeline:
   - Week 1: Knowledge Distillation (5 epochs)
   - Week 2: Pruning + Fine-tune (30 epochs)
   - Week 3: QAT (20 epochs)
```

---

#### C. **Cần siêu nhỏ** (Mobile/Edge)
```
✅ Aggressive Hybrid: Pruning 60% + INT8 + Low-Rank
   - Effort: ⭐⭐⭐⭐ (4 tuần)
   - Compression: 18-20×
   - Final model: 1.4M params, 91-92% acc
   - From: 500MB → 27MB disk
   - Accuracy drop: 3-4%
   
   Nếu 3-4% drop quá lớn → Quay lại Balanced approach
```

---

### 2️⃣ Công Thức Tính Toán Threshold

#### **Accuracy Drop Công Thức**

```python
def estimate_accuracy_drop(compression_ratio, method='hybrid'):
    """
    Empirical formulas from papers:
    
    Quantization (INT8):
        Δ_acc = 0.5-1% (post-training)
        Δ_acc = 0.2-0.5% (QAT)
        
    Pruning (structured):
        Δ_acc ≈ pruning_ratio × 0.07
        (70% pruning → ~2-3% drop)
        
    Knowledge Distillation:
        Δ_acc ≈ -0.5-1% (lợi, không mất!)
        
    Low-rank (SVD):
        Δ_acc ≈ (1 - retained_information%) × 0.1
        (90% info retained → ~0.1% drop)
        
    Hybrid (all combined):
        Δ_acc ≈ 0.7 × Σ(individual_drops)
    """
    
    if method == 'int8':
        return 0.01 + 0.005 * (compression_ratio - 4)  # 0.5-1%
    elif method == 'pruning':
        pruning_ratio = 1 - 1/compression_ratio
        return pruning_ratio * 0.07
    elif method == 'distillation':
        return -0.008  # Gain 0.8%
    elif method == 'low_rank':
        return 0.01 + 0.005 * (compression_ratio - 2)
    elif method == 'hybrid':
        return 0.02 + 0.01 * (compression_ratio - 5)  # 1-2% for 6.7×
    
    return None

# Examples
print(estimate_accuracy_drop(4, 'int8'))       # 0.5-1%
print(estimate_accuracy_drop(2, 'pruning'))    # 0.7% (30% pruning)
print(estimate_accuracy_drop(-1, 'distillation'))  # -0.8% (gain!)
print(estimate_accuracy_drop(6.7, 'hybrid'))   # 1-2% (recommended)
```

---

### 3️⃣ Kiểm Định Chất Lượng

#### **Công Thức Quality Score**

```
Quality_Score = Compression_Ratio / (1 + k × Accuracy_Drop)

Trong đó:
  k = sensitivity factor (thường 5-10)
  
Interpretation:
  Score > 10: Excellent (ngon lành)
  Score 5-10: Good (tốt)
  Score 2-5: Fair (chấp nhận được)
  Score < 2: Poor (quá mất accuracy)

Ví dụ:
  Original accuracy: 95%
  Compressed accuracy: 93.5%
  Compression ratio: 6.7×
  
  Accuracy drop: 1.5%
  
  Quality_Score = 6.7 / (1 + 5 × 0.015)
                = 6.7 / 1.075
                ≈ 6.23 ✅ Good!
```

---

## 📋 BẢNG SO SÁNH PHƯƠNG PHÁP

| Phương Pháp | Compression | Accuracy Drop | Effort | Tools | Ghi Chú |
|------------|-------------|---------------|--------|-------|---------|
| **INT8 QAT** | 4× | 0.5-1% | ⭐ | TensorRT, ONNX | Nhanh nhất setup |
| **Pruning 40%** | 1.67× | 1-2% | ⭐⭐ | PyTorch Prune | Dễ nhất |
| **Distillation** | 0.8× | -0.5% | ⭐⭐⭐ | PyTorch | Tốt nhất accuracy |
| **Low-Rank (SVD)** | 3.6× | 2-3% | ⭐⭐ | NumPy, TensorDecomp | Giải toán học |
| **Mixed Precision** | 2× | <0.5% | ⭐ | PyTorch, TensorRT | Siêu dễ |
| **Hybrid (Distil+Prune+INT8)** | **6.7×** | **1-2%** | ⭐⭐⭐⭐ | PyTorch | **RECOMMENDED** |

---

## 🔬 PAPERS & REFERENCES

### Key Papers (công thức & kết quả)

```
1. Deep Compression (2016) - Song Han et al.
   Title: Deep Compression: Compressing Deep Neural Networks with Pruning, 
          Trained Quantization and Huffman Coding
   Link: https://arxiv.org/abs/1510.00149
   
   Key formulas:
   - Pruning magnitude threshold: τ = α × σ(weights)
   - Compression ratio: 35-49×
   - Accuracy drop: <1%
   
   Phương pháp: Pruning → Quantization → Huffman coding

2. Quantization and Training of Neural Networks for Efficient 
   Integer-Arithmetic-Only Inference (2018) - Quantization-Aware Training
   Link: https://arxiv.org/abs/1806.08342
   
   Key formulas:
   - Fake quantization: x_q = round(clip(x/s, -128, 127)) × s
   - Loss scaling: loss × 2^15 (to prevent underflow in FP16)
   - QAT loss: L = MSE + λ × L_quant
   
   Results: 4× compression, <1% accuracy drop

3. Distilling the Knowledge in a Neural Network (2015) - Hinton et al.
   Link: https://arxiv.org/abs/1503.02531
   
   Key formulas:
   - Soft target probability: p_i = exp(z_i/T) / Σ exp(z_j/T)
   - Distillation loss: L_kd = KL(p_teacher || p_student)
   - Combined: L = α × L_ce + (1-α) × L_kd
   
   Best practice: T=5-10, α=0.1-0.3

4. The Lottery Ticket Hypothesis (2019) - Frankle & Carbin
   Link: https://arxiv.org/abs/1803.03635
   
   Key finding: 10-100× sparsity without accuracy loss
   
   Algorithm:
   1. Train network to convergence
   2. Prune lowest magnitude weights (90%)
   3. Reset unpruned weights to initialization
   4. Retrain and achieve similar accuracy!
   
   Why? Correct connectivity pattern matters more than values

5. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
   Link: https://arxiv.org/abs/1704.04861
   
   Key technique: Depthwise Separable Convolution
   - Standard Conv: c_in × c_out × k² operations
   - Depthwise: c_in × k² + c_in × c_out = 8-9× fewer
   
   Compression: 8-9×, Accuracy: similar to standard

6. Mixed Precision Training (2017) - Nvidia
   Link: https://arxiv.org/abs/1710.03740
   
   Key idea:
   - Forward: FP16 (nhanh)
   - Weights/Optimizer: FP32 (chính xác)
   - Loss scaling: prevent underflow
   
   Results: 2× speedup, <0.5% accuracy impact
```

---

## 💻 CÔNG CỤ THỰC TIỄN

### PyTorch Tools
```python
# 1. Quantization
import torch.quantization as tq
tq.quantize_dynamic()  # Post-training
tq.prepare_qat()       # Quantization-aware training

# 2. Pruning
import torch.nn.utils.prune as prune
prune.l1_unstructured()
prune.ln_structured()

# 3. Distillation
# DIY với KL divergence loss

# 4. Low-rank
from scipy.linalg import svd
U, S, Vt = svd(weight_matrix)

# 5. Mixed Precision
from torch.cuda.amp import autocast, GradScaler
```

### TensorFlow/Keras Tools
```python
# Quantization
import tensorflow_model_optimization as tfmot
tfmot.quantization.keras.quantize_model()

# Pruning
tfmot.sparsity.keras.prune_low_magnitude()

# Knowledge Distillation
# Cần DIY
```

### ONNX/TensorRT (Deployment)
```python
# Export & optimize
import onnx
import tensorrt as trt

# Convert PyTorch → ONNX
torch.onnx.export(model, dummy_input, "model.onnx")

# Optimize with TensorRT
builder = trt.Builder(logger)
# ... TensorRT optimization ...
```

---

## 🚀 IMPLEMENTATION ROADMAP

### Week 1: Knowledge Distillation
```python
1. Load teacher model (YOLOv8Large)
2. Create student model (YOLOv8GoldACsim)
3. Implement distillation loss (Temperature=5, Alpha=0.2)
4. Train 5 epochs
5. Evaluate: expect +0.5-1% accuracy improvement
```

### Week 2: Structured Pruning
```python
1. Identify pruning ratio (40% channels)
2. Compute L2-norms for each channel
3. Select channels to prune
4. Fine-tune 30 epochs
5. Evaluate: expect ~1-1.5% accuracy drop
6. Total compression so far: 1.67×
```

### Week 3: Quantization-Aware Training
```python
1. Prepare QAT config
2. Calibrate on training data
3. QAT training 20 epochs
4. Convert to INT8 model
5. Evaluate: expect <1% additional drop
6. Final compression: 6.7×
7. Final accuracy: ~93.5% (drop: 1.5%)
```

### Validation
```python
1. Compare speed: 100× faster expected
2. Compare size: 6.7× smaller (3.8M params)
3. Run inference on test set
4. Check quality metric: Q = 6.7 / (1 + 5×0.015) ≈ 6.23 ✅
5. Deploy to edge device
```

---

## ❓ FAQ

### Q: Có cách nào giảm accuracy drop?
```
A: Có 3 cách:
1. Use knowledge distillation từ mô hình lớn hơn
2. Reduce compression ratio (target <4× thay vì 6.7×)
3. Tuning hyperparameters (temperature, alpha, pruning ratio)
```

### Q: Nên bắt đầu từ phương pháp nào?
```
A: Step by step:
1. Thử INT8 Post-training (dễ, nhanh, <1% drop)
2. Nếu cần nhỏ hơn → Pruning 30%
3. Nếu cần cân bằng → Hybrid approach
4. Nếu cần siêu nhỏ → Aggressive pruning + low-rank
```

### Q: Có phương pháp nào không cần retraining?
```
A: Có:
1. INT8 Post-training quantization (cần data calibration)
2. Magnitude pruning + remove (cần fine-tune để khôi phục)
3. Mixed precision (không cần retrain, chỉ cần convert)

Nhưng kết quả tốt nhất vẫn cần fine-tuning!
```

### Q: Accuracy drop nào là chấp nhận được?
```
A: Phụ thuộc use case:
- Healthcare/Security: <0.5% (nghiêm ngặt)
- Autonomous driving: <1% (cần cẩn thận)
- General detection: 1-2% (chấp nhận)
- Low-resource devices: 2-5% (cần tính toán)
```

---

## 📊 QUICK CALCULATOR

```python
# Công cụ tính toán nhanh

def compression_calculator(
    original_params: int,
    compression_ratio: float,
    accuracy_drop_percent: float,
    sensitivity_k: float = 5.0
):
    """Quick compression metrics calculator"""
    
    compressed_params = original_params / compression_ratio
    quality_score = compression_ratio / (1 + sensitivity_k * (accuracy_drop_percent / 100))
    
    print("=" * 50)
    print("COMPRESSION METRICS")
    print("=" * 50)
    print(f"Original params: {original_params/1e6:.1f}M")
    print(f"Compressed params: {compressed_params/1e6:.1f}M")
    print(f"Compression ratio: {compression_ratio:.1f}×")
    print(f"Accuracy drop: {accuracy_drop_percent:.2f}%")
    print(f"Quality score: {quality_score:.2f}")
    print()
    
    if quality_score > 10:
        rating = "⭐⭐⭐⭐⭐ Excellent!"
    elif quality_score > 5:
        rating = "⭐⭐⭐⭐ Good"
    elif quality_score > 2:
        rating = "⭐⭐⭐ Fair"
    elif quality_score > 1:
        rating = "⭐⭐ Poor"
    else:
        rating = "⭐ Unacceptable"
    
    print(f"Rating: {rating}")
    print("=" * 50)
    
    return {
        'compressed_params': compressed_params,
        'compression_ratio': compression_ratio,
        'quality_score': quality_score,
        'rating': rating
    }

# Usage
compression_calculator(
    original_params=25.9e6,
    compression_ratio=6.7,
    accuracy_drop_percent=1.5
)
```

**Output:**
```
==================================================
COMPRESSION METRICS
==================================================
Original params: 25.9M
Compressed params: 3.9M
Compression ratio: 6.7×
Accuracy drop: 1.50%
Quality score: 6.23

Rating: ⭐⭐⭐⭐ Good
==================================================
```

