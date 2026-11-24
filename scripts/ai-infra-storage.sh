# Activated by Conda; user-created caches and temporary files stay on Windows.
AI_INFRA_STORAGE='/mnt/e/Visual Studio code/2026_code/vllm'
export XDG_CACHE_HOME="$AI_INFRA_STORAGE/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch-extensions"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export CONDA_PKGS_DIRS="$XDG_CACHE_HOME/conda-pkgs"
export TMPDIR="$XDG_CACHE_HOME/tmp"
unset AI_INFRA_STORAGE
