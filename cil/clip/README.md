# CIL

## Train stage

Example:

```bash
cd cil

# This has 2 initial classes and 2 incremental classes
# Save log to file
bash run_cifar100-2-2.sh 2>&1 | tee train-adapters-"$(date +%y%m%d_%H%M%S)".log
```
