# cuda setting
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# vgl setting
export PATH=/usr/NX/scripts/vgl:$PATH
# 反向代理, 参见 https://www.cnblogs.com/coldchair/p/18050457
# export http_proxy=http://127.0.0.1:10310
# export https_proxy=http://127.0.0.1:10310

# 使用 PROMPT_COMMAND 更新提示符，避免重复显示
PROMPT_COMMAND='if [[ "$CONDA_DEFAULT_ENV" != "" ]]; then \
PS1="\[\033[01;32m\][\t]\[\033[00m\] \
\[\033[38;5;222m\](${CONDA_DEFAULT_ENV})\[\033[00m\] \
\[\033[01;32m\]\u@\h\[\033[00m\]:\
\[\033[01;34m\]\w\[\033[00m\]$ "; \
else \
PS1="\[\033[01;32m\][\t]\[\033[00m\] \
\[\033[01;32m\]\u@\h\[\033[00m\]:\
\[\033[01;34m\]\w\[\033[00m\]$ "; \
fi'
