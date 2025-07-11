# Extending the Model

## Modifying SAF Parameters
Adjust SAF behavior by editing `config.yaml`:
```yaml
# Example SAF configuration (to be implemented)
saf:
  attention_heads: 4
  fusion_method: weighted_average
```

## Adding New Modules
1. Create new module in `model.py`
2. Update `DualModel` class to include your module
3. Adjust forward pass logic