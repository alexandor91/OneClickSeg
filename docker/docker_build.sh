echo "nvidia官方docker, 找匹配的: https://hub.docker.com/r/nvidia/cuda/tags?page=1"

# 指定参数
echo "使用命令行参数:\nUSER_NAME=$1\nCUDA=$2\nCUDNN=$3\nUBUNTU=$4"
USER_NAME=$1
CUDA=$2
CUDNN=$3
UBUNTU=$4

IMAGE_TAG="${USER_NAME}/cuda:${CUDA}_cudnn${CUDNN}_ubuntu${UBUNTU}"     # 改成你的dockerhub名称
echo "构建的镜像: ${IMAGE_TAG}"
echo "base镜像: nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu${UBUNTU}"

# 设置最大重试次数
MAX_RETRIES=10
retry_count=0

build_success=false
while [ $retry_count -lt $MAX_RETRIES ] && [ "$build_success" = false ]; do
    echo "尝试构建 Docker 镜像 (尝试 $((retry_count+1))/$MAX_RETRIES)..."
    
    docker build . --tag ${IMAGE_TAG}\
        --build-arg CUDA=${CUDA} \
        --build-arg CUDNN=${CUDNN} \
        --build-arg UBUNTU=${UBUNTU}
    
    if [ $? -eq 0 ]; then
        build_success=true
        echo "Docker 镜像 '${IMAGE_TAG}' 构建成功！"
    else
        retry_count=$((retry_count+1))
        if [ $retry_count -lt $MAX_RETRIES ]; then
            echo "构建失败，将在 5 秒后重试..."
            sleep 5
        fi
    fi
done

if [ "$build_success" = false ]; then
    echo "在 $MAX_RETRIES 次尝试后, Docker 镜像 '${IMAGE_TAG}' 构建仍然失败"
    exit 1
fi

exit 0