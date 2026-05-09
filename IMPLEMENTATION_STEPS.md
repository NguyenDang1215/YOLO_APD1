# 🛠️ IMPLEMENTATION GUIDE: Nén YOLOv8 GoldACsim

## 📌 TL;DR (Tóm Tắt)

```
Mục tiêu: Nén từ 25.9M → 3.8M params (6.7× nhỏ hơn)
         Giữ accuracy 93.5% (drop: 1.5%)
         
Phương pháp: Knowledge Distillation + Pruning + QAT
Thời gian: 3-4 tuần
Công cụ: PyTorch, TensorRT
```

---

## 🎬 STEP-BY-STEP IMPLEMENTATION

### PHASE 1: Knowledge Distillation (1 tuần)

#### Bước 1.1: Chuẩn Bị

```python
# install_requirements.txt
torch>=2.0.0
torchvision>=0.15.0
tensorrt>=8.6.0
onnx>=1.14.0
onnxruntime>=1.16.0
```

#### Bước 1.2: Load Models

```python
# compression_pipeline.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

# Load teacher (large model)
teacher = YOLO('yolov8l.pt')
teacher_model = teacher.model
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False

# Load student (our model)
student = YOLO('path/to/yolo8_goldacsim.pt')
student_model = student.model
student_model.train()
```

#### Bước 1.3: Distillation Loss

```python
class DistillationLoss(nn.Module):
    def __init__(self, temperature=5.0, alpha=0.2):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, targets):
        """
        Công thức:
        L_total = α × L_CE + (1-α) × T² × L_KL
        
        Args:
            student_logits: (B, 85×num_anchors) - output từ student
            teacher_logits: (B, 85×num_anchors) - output từ teacher
            targets: ground truth
        """
        # Cross-entropy loss (học từ ground truth)
        ce = self.ce_loss(student_logits, targets)
        
        # Knowledge distillation loss (học từ teacher)
        # Soft probabilities với temperature
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        
        # Combine
        loss = self.alpha * ce + (1 - self.alpha) * (self.temperature ** 2) * kl_loss
        return loss, ce.item(), kl_loss.item()

# Test loss
distill_loss = DistillationLoss(temperature=5.0, alpha=0.2)
```

#### Bước 1.4: Training Loop

```python
def train_distillation(teacher_model, student_model, train_loader, 
                       val_loader, epochs=5, device='cuda'):
    """
    Knowledge Distillation Training
    
    Timeline: 1 epoch ~ 1 hour (với 100k steps)
    Total: ~5 hours cho 5 epochs
    """
    
    teacher_model.eval()
    student_model.train()
    
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=0.001,
        weight_decay=0.0005
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs * len(train_loader)
    )
    
    distill_loss_fn = DistillationLoss(temperature=5.0, alpha=0.2)
    
    best_acc = 0
    history = {'ce_loss': [], 'kl_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs} - Knowledge Distillation")
        print(f"{'='*60}")
        
        total_ce_loss = 0
        total_kl_loss = 0
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            with torch.no_grad():
                teacher_out = teacher_model(images)
                # Extract features từ teacher
                teacher_features = teacher_out[0]  # Adjust based on actual output
            
            student_out = student_model(images)
            student_features = student_out[0]
            
            # Loss
            loss, ce, kl = distill_loss_fn(student_features, teacher_features, targets)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_ce_loss += ce
            total_kl_loss += kl
            num_batches += 1
            
            if (batch_idx + 1) % 100 == 0:
                avg_ce = total_ce_loss / num_batches
                avg_kl = total_kl_loss / num_batches
                print(f"Batch {batch_idx+1}: CE Loss={avg_ce:.4f}, KL Loss={avg_kl:.4f}")
        
        # Validation
        val_acc = evaluate(student_model, val_loader, device)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  CE Loss: {total_ce_loss/num_batches:.4f}")
        print(f"  KL Loss: {total_kl_loss/num_batches:.4f}")
        print(f"  Val Accuracy: {val_acc:.2f}%")
        
        history['ce_loss'].append(total_ce_loss/num_batches)
        history['kl_loss'].append(total_kl_loss/num_batches)
        history['val_acc'].append(val_acc)
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student_model.state_dict(), 
                      'checkpoint_distilled_best.pt')
            print(f"  ✅ New best accuracy: {val_acc:.2f}%")
    
    return student_model, history

# Train
student_model, history = train_distillation(
    teacher_model, 
    student_model,
    train_loader,
    val_loader,
    epochs=5,
    device='cuda'
)

# Expected result: Accuracy ≈ 95% (improved by ~0.5-1%)
print("Phase 1 Complete: Knowledge Distillation ✅")
```

---

### PHASE 2: Structured Pruning (1-1.5 tuần)

#### Bước 2.1: Tính Channel Importance

```python
def compute_channel_importance(model):
    """
    Công thức:
    importance[i] = L2_norm(weights[i, :, :, :])
    
    Lý do: Channels có L2-norm nhỏ → ít ảnh hưởng
    """
    
    importance_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            w = module.weight.data  # Shape: (out_c, in_c, k, k)
            
            # Compute L2-norm cho mỗi output channel
            norms = torch.norm(w.view(w.size(0), -1), p=2, dim=1)
            
            importance_dict[name] = {
                'norms': norms,
                'out_channels': w.size(0),
                'in_channels': w.size(1),
                'kernel': w.size(2)
            }
    
    return importance_dict

importance = compute_channel_importance(student_model)
```

#### Bước 2.2: Magnitude-based Pruning

```python
class StructuredPruner:
    def __init__(self, model, pruning_ratio=0.4):
        """
        pruning_ratio: tỷ lệ channels bị xóa (0.4 = 40%)
        """
        self.model = model
        self.ratio = pruning_ratio
        self.masks = {}
    
    def compute_masks(self):
        """
        Compute pruning masks dựa trên magnitude
        """
        importance = compute_channel_importance(self.model)
        
        for name, info in importance.items():
            norms = info['norms']
            out_c = info['out_channels']
            
            # Tính ngưỡng: giữ (1-ratio) channels có norm cao nhất
            k = int(out_c * (1 - self.ratio))
            threshold = torch.topk(norms, k, largest=True)[0][-1]
            
            # Mask: 1 = giữ, 0 = xóa
            mask = (norms >= threshold).float()
            
            self.masks[name] = mask
            
            print(f"{name}: Keep {mask.sum().item()}/{out_c} channels")
        
        return self.masks
    
    def apply_masks(self):
        """
        Áp dụng masks vào model
        """
        for name, module in self.model.named_modules():
            if name in self.masks:
                mask = self.masks[name].to(module.weight.device)
                
                # Zero out những channels bị prune
                # Thực tế này là "soft pruning", cần thay đổi architecture
                # để "hard prune" (thực sự giảm memory)
                
                with torch.no_grad():
                    module.weight.data = module.weight.data * mask.view(-1, 1, 1, 1)

pruner = StructuredPruner(student_model, pruning_ratio=0.4)
masks = pruner.compute_masks()
pruner.apply_masks()
```

#### Bước 2.3: Fine-tune Pruned Model

```python
def train_pruned_model(model, train_loader, val_loader, 
                       epochs=30, device='cuda'):
    """
    Fine-tune sau pruning
    
    Công thức:
    L = CE_loss + λ × L_magnitude_reg
    
    (để tránh pruned channels "bật lại")
    """
    
    model.train()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0005,  # Lower learning rate
        weight_decay=0.0005
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs} - Pruning Fine-tune")
        
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(images)
            
            # Loss
            ce_loss = criterion(outputs, targets)
            
            # Regularization: Magnitude penalty để tránh pruned channels hiện lại
            magnitude_loss = 0
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) and name in masks:
                    mask = masks[name].to(module.weight.device)
                    # Penalty nếu pruned channels có magnitude lớn
                    pruned_weights = module.weight.data * (1 - mask.view(-1, 1, 1, 1))
                    magnitude_loss += torch.norm(pruned_weights)
            
            total_loss_batch = ce_loss + 0.001 * magnitude_loss
            
            # Backward
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Batch {batch_idx+1}: Loss={total_loss/num_batches:.4f}")
        
        # Validation
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} - Loss: {total_loss/num_batches:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'checkpoint_pruned_best.pt')
            print(f"✅ New best accuracy: {val_acc:.2f}%")
    
    return model

# Fine-tune
student_model = train_pruned_model(
    student_model,
    train_loader,
    val_loader,
    epochs=30
)

# Expected: Accuracy ≈ 93-94% (drop: 1-2%)
print("Phase 2 Complete: Structured Pruning ✅")
```

---

### PHASE 3: Quantization-Aware Training (1 tuần)

#### Bước 3.1: Chuẩn Bị QAT

```python
import torch.quantization as tq

def prepare_qat_model(model):
    """
    Chuẩn bị model cho QAT
    
    Công thức:
    Forward: x_q = round(clip(x/scale, -128, 127)) × scale
    Backward: ∂L/∂x ≈ ∂L/∂x_q (straight-through estimator)
    """
    
    # Backend: 'fbgemm' cho CPU, 'qnnpack' cho ARM
    model.qconfig = tq.get_default_qat_qconfig('fbgemm')
    
    # Prepare model
    tq.prepare_qat(model, inplace=True)
    
    return model

student_model = prepare_qat_model(student_model)
```

#### Bước 3.2: QAT Training

```python
def train_qat_model(model, train_loader, val_loader, 
                    epochs=20, device='cuda'):
    """
    Quantization-Aware Training
    
    Timeline: ~2 hours cho 20 epochs
    """
    
    model.train()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0001,  # Very low learning rate
        weight_decay=0.0005
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs * len(train_loader)
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Loss scaling để tránh underflow trong FP16/INT8
    loss_scale = 2 ** 15  # 32768
    
    best_acc = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs} - Quantization-Aware Training")
        
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward (INT8 simulation)
            outputs = model(images)
            
            # Loss
            loss = criterion(outputs, targets)
            
            # Loss scaling
            loss_scaled = loss * loss_scale
            
            # Backward
            loss_scaled.backward()
            
            # Unscale gradients
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data = param.grad.data / loss_scale
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Batch {batch_idx+1}: Loss={total_loss/num_batches:.4f}")
        
        # Validation
        model.eval()
        val_acc = evaluate(model, val_loader, device)
        model.train()
        
        print(f"Epoch {epoch+1} - Loss: {total_loss/num_batches:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            # Save before convert
            torch.save(model.state_dict(), 'checkpoint_qat_best.pt')
            print(f"✅ New best accuracy: {val_acc:.2f}%")
    
    return model

# Train QAT
student_model = train_qat_model(
    student_model,
    train_loader,
    val_loader,
    epochs=20
)

# Convert to INT8
student_model = tq.convert(student_model, inplace=True)

# Expected: Accuracy ≈ 93.5% (total drop: 1.5% from 95%)
print("Phase 3 Complete: Quantization ✅")
```

---

### PHASE 4: Validation & Deployment

#### Bước 4.1: Validation

```python
def validate_compressed_model(original_model, compressed_model, 
                              val_loader, device='cuda'):
    """
    Validate compressed model quality
    """
    
    original_model.eval()
    compressed_model.eval()
    
    original_acc = evaluate(original_model, val_loader, device)
    compressed_acc = evaluate(compressed_model, val_loader, device)
    
    accuracy_drop = original_acc - compressed_acc
    
    # Model sizes
    original_size = sum(p.numel() for p in original_model.parameters())
    compressed_size = sum(p.numel() for p in compressed_model.parameters())
    compression_ratio = original_size / compressed_size
    
    # Quality score
    quality_score = compression_ratio / (1 + 5 * (accuracy_drop / 100))
    
    print("="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(f"Original accuracy: {original_acc:.2f}%")
    print(f"Compressed accuracy: {compressed_acc:.2f}%")
    print(f"Accuracy drop: {accuracy_drop:.2f}%")
    print()
    print(f"Original params: {original_size/1e6:.1f}M")
    print(f"Compressed params: {compressed_size/1e6:.1f}M")
    print(f"Compression ratio: {compression_ratio:.1f}×")
    print()
    print(f"Quality score: {quality_score:.2f}")
    
    if quality_score > 5:
        print("✅ ACCEPTABLE - Approved for deployment!")
    else:
        print("❌ NOT ACCEPTABLE - Need more tuning")
    print("="*60)
    
    return {
        'original_acc': original_acc,
        'compressed_acc': compressed_acc,
        'accuracy_drop': accuracy_drop,
        'compression_ratio': compression_ratio,
        'quality_score': quality_score
    }

# Validate
results = validate_compressed_model(
    teacher_model,  # Use original as reference
    student_model,
    val_loader
)
```

#### Bước 4.2: Export to ONNX

```python
def export_to_onnx(model, dummy_input, output_path='model.onnx'):
    """
    Export compressed model to ONNX format
    """
    
    model.eval()
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=True
    )
    
    print(f"✅ Model exported to {output_path}")

# Export
dummy_input = torch.randn(1, 3, 640, 640).to(device)
export_to_onnx(student_model, dummy_input, 'yolo8_compressed.onnx')
```

#### Bước 4.3: Deploy

```python
def deploy_with_tensorrt(onnx_path, device_id=0):
    """
    Deploy với TensorRT cho maximum speed
    """
    
    import tensorrt as trt
    
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX
    with open(onnx_path, 'rb') as model_file:
        parser.parse(model_file.read())
    
    # Build engine
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)  # Enable INT8
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    
    engine = builder.build_serialized_network(network, config)
    
    print(f"✅ TensorRT engine built: {engine}")
    
    return engine

# Deploy
engine = deploy_with_tensorrt('yolo8_compressed.onnx')
```

---

## 📊 EXPECTED RESULTS

```
Timeline:
├─ Week 1: Knowledge Distillation
│  └─ Accuracy: 95% → 95.5% (gain +0.5%)
│  └─ Model size: unchanged
│  └─ Time: ~5 hours training
│
├─ Week 2: Pruning
│  └─ Accuracy: 95.5% → 94.3% (drop -1.2%)
│  └─ Model size: 25.9M → 15.5M (60% retained)
│  └─ Compression so far: 1.67×
│  └─ Time: ~5 hours fine-tune
│
├─ Week 3: QAT
│  └─ Accuracy: 94.3% → 93.5% (drop -0.8%)
│  └─ Model size: 15.5M → 3.9M (INT8)
│  └─ Total compression: 6.7×
│  └─ Time: ~2 hours training
│
└─ Week 4: Deployment
   └─ Exported to ONNX
   └─ TensorRT optimized
   └─ Ready for edge devices

FINAL METRICS:
- Original: 25.9M params, 95% accuracy, 500MB
- Compressed: 3.9M params, 93.5% accuracy, 77MB
- Compression: 6.7×
- Accuracy drop: 1.5% (acceptable)
- Speed gain: 100-300×
```

---

## ✅ CHECKLIST

```
□ Week 1: Knowledge Distillation
  □ Load teacher & student models
  □ Implement distillation loss
  □ Train 5 epochs
  □ Validate accuracy improvement
  □ Save checkpoint

□ Week 2: Structured Pruning
  □ Compute channel importance (L2-norm)
  □ Create pruning masks (40% ratio)
  □ Apply masks
  □ Fine-tune 30 epochs
  □ Validate accuracy drop < 1.5%
  □ Save checkpoint

□ Week 3: Quantization-Aware Training
  □ Prepare QAT config
  □ Train 20 epochs
  □ Convert to INT8
  □ Validate accuracy drop < 0.8%
  □ Save checkpoint

□ Week 4: Deployment
  □ Validate overall compression
  □ Export to ONNX
  □ Build TensorRT engine
  □ Test on edge device
  □ Benchmark speed/accuracy
  □ Deploy to production
```

