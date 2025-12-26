###############################################################################
# DeepSpeed fused_adam build fix (no admin): make CUDA headers/libs discoverable
# Paste this whole block in your run.ai terminal BEFORE running deepspeed.
###############################################################################

# 0) Use the container CUDA compiler
export CUDA_HOME=/opt/conda
export PATH="$CUDA_HOME/bin:$PATH"

# 1) Build PyTorch/DeepSpeed JIT extensions on local disk (avoid RDS/NFS issues)
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_extensions"
mkdir -p "$TORCH_EXTENSIONS_DIR"

# Optional: clear failed fused_adam builds to force a clean rebuild
# rm -rf "$TORCH_EXTENSIONS_DIR"/fused_adam* "$HOME/.cache/torch_extensions"/fused_adam* 2>/dev/null || true

# 2) Add ALL CUDA component headers provided via python nvidia/* packages
#    (covers: cusparse, cublas, cusolver, curand, etc.)
for d in /opt/conda/lib/python3.12/site-packages/nvidia/*/include; do
  [ -d "$d" ] && export CPATH="$d:$CPATH" \
               && export C_INCLUDE_PATH="$d:$C_INCLUDE_PATH" \
               && export CPLUS_INCLUDE_PATH="$d:$CPLUS_INCLUDE_PATH"
done

# 3) Add ALL CUDA component libraries provided via python nvidia/* packages
for d in /opt/conda/lib/python3.12/site-packages/nvidia/*/lib; do
  [ -d "$d" ] && export LIBRARY_PATH="$d:$LIBRARY_PATH" \
               && export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
done

# Also keep /opt/conda/lib in search paths (harmless, often useful)
export LIBRARY_PATH="/opt/conda/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/opt/conda/lib:$LD_LIBRARY_PATH"

# 4) (Optional) Reduce compile parallelism if your environment is tight on RAM
# export MAX_JOBS=4

# 5) Quick sanity prints (optional; comment out if you want it quieter)
# echo "CUDA_HOME=$CUDA_HOME"
# which nvcc
# nvcc --version | head -n 2
# echo "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"

###############################################################################
# Now run your training
# deepspeed d4/train.py --deepspeed_config d4/training/ds_config.json
###############################################################################
