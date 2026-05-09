#!/bin/bash
# 本脚本允许 nohup 执行
# 等待指定几个gpu上指定进程结束，迭代进行训练任务, 无限循环
# 参数：
# -g: 指定gpu编号，多个gpu用逗号分隔
# -e: conda 环境名
# -d, -c, -n: 训练任务相关参数, 数据集名、配置文件名、保存文件夹名
# 检测gpu上非Xorg的进程，获取进程号，并等待进程结束

usage() {
    echo "Usage:"
    echo "export QQ_EMAIL_USER=<你的QQ邮箱@qq.com>"
    echo "export QQ_EMAIL_PASS=<你的授权码>"
    echo "nohup bash $0 -g <gpu_ids> -e <conda_env> -d <dataset> -c <config> -n <save_dir> > /dev/null 2>&1 &"
    exit 1
}

while getopts "g:e:d:c:n:" opt; do
  case $opt in
    g) GPU_IDS=$OPTARG ;;
    e) CONDA_ENV=$OPTARG ;;
    d) DATASET=$OPTARG ;;
    c) CONFIG=$OPTARG ;;
    n) SAVE_DIR=$OPTARG ;;
    *) usage ;;
  esac
done

if [[ -z "$GPU_IDS" || -z "$CONDA_ENV" || -z "$DATASET" || -z "$CONFIG" || -z "$SAVE_DIR" ]]; then
    usage
fi

IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"

# 获取当前机器名作为标题, 发送邮件
python tools/send_mail.py -t $QQ_EMAIL_USER \
    -s "[$(hostname)] Waiting for GPU(s) $GPU_IDS to be free..." \
    -b "<b>[$(hostname)]:</b> -g ${#GPU_ARRAY[@]} -d $DATASET -c $CONFIG -n $SAVE_DIR<br><p><strong>时间:</strong> $(date)</p>"

while true; do
    BUSY=0
    for gpu in "${GPU_ARRAY[@]}"; do
        # 检查该GPU上是否有非Xorg进程
        pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $gpu)
        for pid in $pids; do
            if [ -n "$pid" ]; then
                pname=$(ps -p $pid -o comm=)
                if [ "$pname" != "Xorg" ]; then
                    BUSY=1
                    break 2
                fi
            fi
        done
    done
    if [ $BUSY -eq 1 ]; then
        echo "[wait_gpu] GPU(s) $GPU_IDS busy, waiting... $(date)"
    else
        echo "[wait_gpu] GPU(s) $GPU_IDS free, starting training... $(date)"
        source ~/.bashrc
        conda activate $CONDA_ENV
        # 不得已, wandb 网络有问题
        export WANDB_MODE=offline
        # nohup 训练命令
        CUDA_VISIBLE_DEVICES=$GPU_IDS nohup sh scripts/train.sh -g ${#GPU_ARRAY[@]} -d $DATASET -c $CONFIG -n $SAVE_DIR > /dev/null 2>&1 &
        # 获取当前机器名作为标题, 发送邮件
        python tools/send_mail.py -t $QQ_EMAIL_USER \
            -s "[$(hostname)] Training task submitted" \
            -b "<b>[$(hostname)]:</b> -g ${#GPU_ARRAY[@]} -d $DATASET -c $CONFIG -n $SAVE_DIR<br><p><strong>时间:</strong> $(date)</p>"
        conda deactivate
    fi
    sleep 60
done
