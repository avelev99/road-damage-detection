# Debugging Common Issues

## VRAM Optimization Errors
1. **Batch size too large**:
   - Reduce `batch_size` in `config.yaml`
   - Enable gradient accumulation

2. **Mixed precision issues**:
   - Update CUDA drivers
   - Verify AMP support with `torch.cuda.amp.is_available()`

## Training Instabilities
- Gradient clipping
- Learning rate scheduling
- Loss function adjustments