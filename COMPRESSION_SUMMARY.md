# 📚 Model Compression - Complete Summary

## 📁 Tệp Tài Liệu Đã Tạo

```
1. MODEL_COMPRESSION_GUIDE.md (15+ pages)
   ├─ Quantization (INT8, QAT, PTQ)
   ├─ Pruning (Structured, Unstructured, Lottery Ticket)
   ├─ Knowledge Distillation
   ├─ Low-Rank Approximation
   ├─ Mixed Precision
   ├─ Hybrid Approaches
   ├─ Công thức tính toán
   └─ 50+ papers & references

2. COMPRESSION_QUICK_GUIDE.md
   ├─ Decision tree (chọn phương pháp)
   ├─ Công thức Quality Score
   ├─ Tools & Libraries
   ├─ Implementation roadmap
   ├─ FAQ
   └─ Quick calculator

3. IMPLEMENTATION_STEPS.md (Step-by-step code)
   ├─ Phase 1: Knowledge Distillation (5 epochs, 5h)
   ├─ Phase 2: Structured Pruning (30 epochs, 5h)
   ├─ Phase 3: QAT (20 epochs, 2h)
   ├─ Phase 4: Deployment (ONNX → TensorRT)
   ├─ Full Python code
   └─ Validation & testing
```

---

## 🎯 RECOMMENDED APPROACH FOR YOLOv8 GoldACsim

### Phương Pháp: Hybrid (Distillation + Pruning + QAT)

```
Original Model:
  - Parameters: 25.9M
  - Accuracy: 95%
  - Disk size: 500MB
  - RAM: ~1.04GB

Target Model:
  - Parameters: 3.9M (6.7× nhỏ hơn)
  - Accuracy: 93.5% (drop: 1.5%)
  - Disk size: 77MB (6.5× nhỏ hơn)
  - RAM: ~156MB (6.7× nhỏ hơn)
  - Speed: 100-300× faster (TensorRT)
```

### Timeline: 3-4 tuần

| Week | Phase | Task | Duration | Effort | Expected Drop |
|------|-------|------|----------|--------|---------------|
| 1 | Distillation | Train 5 epochs | 5h | ⭐ | -0.5% (gain!) |
| 2 | Pruning 40% | Fine-tune 30 epochs | 5h | ⭐⭐ | -1.2% |
| 3 | QAT | Train 20 epochs | 2h | ⭐ | -0.8% |
| 4 | Deployment | ONNX → TensorRT | 8h | ⭐⭐ | - |

**Total accuracy drop: 1.5%** ✅ (chấp nhận được)

---

## 🔬 CÔNG THỨC CHÍNH

### 1️⃣ Knowledge Distillation

```
Loss = α × CE_loss + (1-α) × T² × KL_divergence

Optimal parameters:
  - Temperature T: 5-7
  - Alpha α: 0.1-0.3
  - Expected improvement: +0.5-1%

KL divergence công thức:
  KL(P||Q) = Σ P(i) × log(P(i)/Q(i))
  
  P = softmax(teacher_logits / T)
  Q = softmax(student_logits / T)
```

### 2️⃣ Structured Pruning

```
Importance formula:
  importance[i] = L2_norm(weight[i, :, :, :])

Pruning ratio:
  ratio = num_pruned_channels / total_channels

Accuracy drop formula:
  Δ_acc ≈ pruning_ratio × 0.07
  
  Example: 40% pruning → 2.8% drop
          30% pruning → 2.1% drop

Compression ratio:
  CR = original_FLOPs / pruned_FLOPs
  CR ≈ 1 / (1 - pruning_ratio)
  
  Example: 40% pruning → 1.67× compression
```

### 3️⃣ Quantization-Aware Training (QAT)

```
Quantization formula:
  q = round(x / scale) + zero_point
  x_approx = (q - zero_point) × scale

Loss scaling (để tránh underflow):
  loss_scaled = loss × 2^15
  grad_unscaled = grad_scaled / 2^15

QAT loss:
  L_total = MSE_loss + λ × quantization_penalty
  
  λ thường = 0.001-0.01

Expected accuracy drop:
  - Post-training QAT: 1-2%
  - QAT training: 0.5-1%
```

### 4️⃣ Quality Score

```
Metric = Compression_Ratio / (1 + k × Accuracy_Drop)

Sensitivity factor k = 5-10

Interpretation:
  Score > 10: Excellent ⭐⭐⭐⭐⭐
  Score 5-10: Good ⭐⭐⭐⭐
  Score 2-5: Fair ⭐⭐⭐
  Score < 2: Poor ⭐⭐

Example (YOLOv8 GoldACsim):
  CR = 6.7×
  Drop = 1.5%
  Score = 6.7 / (1 + 5×0.015) = 6.23 ✅ Good!
```

---

## 📊 SO SÁNH CÁC PHƯƠNG PHÁP

```
┌──────────────┬────────────┬──────────┬──────────┬─────────┐
│ Phương Pháp  │ Compression│ Accuracy │  Effort  │  Notes  │
│              │   Ratio    │   Drop   │          │         │
├──────────────┼────────────┼──────────┼──────────┼─────────┤
│ INT8 QAT     │    4×      │ 0.5-1%   │  ⭐⭐    │  Nhanh  │
│              │            │          │          │ setup   │
├──────────────┼────────────┼──────────┼──────────┼─────────┤
│ Pruning 40%  │   1.67×    │ 1-2%     │  ⭐      │ Dễ nhất │
│              │            │          │          │         │
├──────────────┼────────────┼──────────┼──────────┼─────────┤
│ Distillation │    0.8×    │ -0.5%    │  ⭐⭐⭐  │ Cải    │
│              │            │ (gain!)  │          │ thiện   │
├──────────────┼────────────┼──────────┼──────────┼─────────┤
│ Low-rank SVD │   3.6×     │ 2-3%     │  ⭐⭐    │ Toán   │
│              │            │          │          │ học    │
├──────────────┼────────────┼──────────┼──────────┼─────────┤
│ Mixed Prec   │    2×      │ <0.5%    │  ⭐      │ Siêu   │
│              │            │          │          │ dễ     │
├──────────────┼────────────┼──────────┼──────────┼─────────┤
│ HYBRID       │   6.7×     │ 1-2%     │  ⭐⭐⭐⭐ │RECOMMENDED
│(Rec.)        │            │          │          │        │
└──────────────┴────────────┴──────────┴──────────┴─────────┘
```

---

## 🔑 KEY INSIGHTS

### 1. Accuracy Drop Không Phải Tuyến Tính

```
Intuition:
  - 30% pruning: ~2% drop (dễ)
  - 50% pruning: ~3% drop (nhưng không phải 5%)
  - 70% pruning: ~4% drop (model học cách "mlem" weights)

Vì sao?
  - Model over-parameterized → dự phòng
  - Correct connectivity pattern quan trọng hơn values
  - Fine-tuning khôi phục được một số "connections"
```

### 2. Phối Hợp Các Kỹ Thuật > Từng Cái

```
Sequential Application:
  1. Distillation: +1% accuracy
  2. Pruning: -1.2% accuracy → Net: -0.2%
  3. QAT: -0.8% accuracy → Net: -1%
  
Result: 6.7× compression với drop 1% (tốt!)

Nếu làm riêng:
  1. Chỉ QAT: 4× compression, 1-2% drop
  2. Chỉ Pruning: 2× compression, 2% drop
  3. Chỉ Distillation: không nén

Kết luận: Phối hợp là bắt buộc!
```

### 3. Temperature trong Distillation Rất Quan Trọng

```
T quá nhỏ (T=1):
  - Hard targets (one-hot)
  - Student học ít information
  - Kết quả: tệ

T thích hợp (T=5-7):
  - Soft targets (smooth distribution)
  - Student học nhiều fine-grained info
  - Kết quả: +0.5-1% improvement

T quá lớn (T=20+):
  - Quá mềm, mất specificity
  - Student học từ noise
  - Kết quả: tệ

Rule of thumb: T = 5 + (dataset_size / 1000000)
```

### 4. Per-Channel vs Per-Layer Quantization

```
Per-tensor quantization:
  Một scale cho cả layer
  - Nhanh (1 value per layer)
  - Kém chính xác (1-2% drop)

Per-channel quantization:
  Một scale cho mỗi output channel
  - Chậm hơn (nhưng still OK)
  - Chính xác hơn (<0.5% drop)
  
Recommendation: Dùng per-channel cho better accuracy
```

---

## 💡 PRACTICAL TIPS

### 1. Gradient Clipping

```
Tại sao?
  - QAT + INT8 → gradients có thể explode
  
Cách làm:
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  
Effect:
  - Cân bằng gradients
  - Tránh overshoot
  - Training ổn định hơn
```

### 2. Learning Rate Scheduler

```
Phase 1 (Distillation):
  - LR: 0.001
  - Scheduler: CosineAnnealingLR (natural decay)
  
Phase 2 (Pruning):
  - LR: 0.0005 (thấp hơn)
  - Scheduler: ReduceLROnPlateau (adaptive)
  - Lý do: Model already pruned, cần tinh chỉnh
  
Phase 3 (QAT):
  - LR: 0.0001 (rất thấp)
  - Scheduler: CosineAnnealingLR
  - Lý do: INT8 nhạy cảm với LR lớn
```

### 3. Batch Size

```
Distillation:
  - Batch size: 32-64 (normal)
  - Vì: Cần variation để student học tốt

Pruning:
  - Batch size: 32-64 (same)
  - Vì: Fine-tuning từ checkpoint

QAT:
  - Batch size: 16-32 (nhỏ hơn)
  - Vì: INT8 sensitive to batch statistics
  - Per-channel quantization need smaller batch

Rule: Nhỏ nhất có thể, nhưng >= 16
```

### 4. Calibration Data

```
Cho quantization, cần calibration dataset:
  - Kích thước: 100-500 samples (suffices)
  - Đại diện: Phải có các scenes khác nhau
  - Không: Không dùng training data (biased)
  
Tại sao?
  - Tìm scale/zero_point optimal
  - Dựa trên activation distribution
  - Training data → quá optimistic estimate
```

---

## ❓ TROUBLESHOOTING

### Problem 1: Accuracy drop > 2% sau Pruning

```
Nguyên nhân:
  1. Pruning ratio quá cao (> 50%)
  2. Learning rate quá lớn
  3. Fine-tune epochs không đủ

Giải pháp:
  1. Giảm pruning ratio: 40% → 30%
  2. Giảm LR: 0.0005 → 0.0001
  3. Tăng epochs: 30 → 50
```

### Problem 2: QAT không converge

```
Nguyên nhân:
  1. Learning rate quá lớn
  2. Batch size quá lớn
  3. Loss scaling sai

Giải pháp:
  1. LR: 0.0001 → 0.00005
  2. Batch: 32 → 16
  3. Loss scale: 2^15 → 2^16
  4. Thêm warmup epochs
```

### Problem 3: INT8 model inference bị lỗi

```
Nguyên nhân:
  1. Quantization không tương thích với layer
  2. Export to ONNX sai
  3. TensorRT chưa build đúng

Giải pháp:
  1. Skip quantize một số layers: model.qconfig = None
  2. Check ONNX opset version (>= 13)
  3. Dùng FP32 calibration, convert to INT8 sau
```

---

## 🎯 DECISION GUIDE

### Bạn cần gì?

```
A. "Chỉ nhanh hơn, không quan tâm size"
   → Mixed Precision (FP16) ✅
   - 2× memory savings
   - 2-4× speed improvement
   - <0.5% accuracy drop
   - Effort: 1 hour

B. "Cần cân bằng speed & size, vẫn accurate"
   → Recommended: Distillation + Pruning + QAT ✅
   - 6.7× compression
   - 100× speed improvement
   - 1.5% accuracy drop
   - Effort: 3 weeks

C. "Siêu nhỏ, acceptable <3% drop"
   → Aggressive Hybrid ✅
   - 18-20× compression
   - 300× speed improvement
   - 3-4% accuracy drop
   - Effort: 4 weeks

D. "Mobile/Embedded device"
   → Distillation + Pruning 40% + QAT + ONNX ✅
   - 6.7× compression
   - Export to ONNX
   - Ready for mobile framework
   - Effort: 3 weeks
```

---

## 📈 EXPECTED RESULTS

```
Baseline (Original YOLOv8 GoldACsim):
  ├─ Model Size: 25.9M params (~500MB disk)
  ├─ Accuracy: 95%
  ├─ Latency: 10ms (GPU)
  └─ Memory: 1.04GB

After Compression (Recommended):
  ├─ Model Size: 3.9M params (~77MB disk)
  ├─ Accuracy: 93.5%
  ├─ Latency: 0.1ms (TensorRT) → 100× faster!
  ├─ Memory: 156MB
  └─ Quality Score: 6.23 ✅

Improvement:
  ├─ 6.7× smaller on disk
  ├─ 6.7× less memory
  ├─ 100× faster inference
  ├─ 1.5% accuracy loss (chấp nhận)
  └─ Ready for production edge devices! 🎉
```

---

## 🔗 RESOURCES

### Tools
- **PyTorch**: torch.quantization, torch.nn.utils.prune
- **TensorRT**: High-performance inference optimization
- **ONNX**: Model interoperability standard
- **TVM**: Compiler for deep learning
- **TensorFlow Lite**: Mobile deployment

### Papers
- Deep Compression (2016)
- Quantization-Aware Training (2018)
- Knowledge Distillation (2015)
- Lottery Ticket Hypothesis (2019)
- MobileNets (2017)
- Mixed Precision Training (2017)

### Communities
- Papers with Code (paperswithcode.com)
- GitHub: pytorch/examples
- ArXiv: arxiv.org/search?query=model+compression

