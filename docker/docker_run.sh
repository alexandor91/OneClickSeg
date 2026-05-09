# 镜像名称 IMAGE
read -p "请输入需要使用的镜像名称(例: yuzijian/cuda:11.6.1_cudnn8_ubuntu20.04): " IMAGE
while test -z "$IMAGE"
do
    read -p "请输入内容为空，请重新输入: " IMAGE
done

# 容器名称 CONTAINER
read -p "请设置容器名称(例: yzj_agile3d): " CONTAINER
while test -z "$CONTAINER"
do
    read -p "输入内容为空，请重新输入: " CONTAINER
done

# Nomachine映射端口 NomachineBindPort
read -p "请设置Nomachine映射端口(例: 14567): " NomachineBindPort
while test -z "$NomachineBindPort"
do
    read -p "输入内容为空，请重新输入: " NomachineBindPort
done

# SSH 映射端口 SshBindPort
read -p "请设置SSH映射端口(例: 10022): " SshBindPort
while test -z "$SshBindPort"
do
    read -p "输入内容为空，请重新输入: " SshBindPort
done

# udp 映射端口 UdpBindPort
read -p "请设置udp映射端口(例: 10093): " UdpBindPort
while test -z "$UdpBindPort"
do
    read -p "输入内容为空，请重新输入: " UdpBindPort
done

# 工作空间映射目录 WorkSpaceBind 
read -p "请设置映射到'/data'的目录(例: /data/shared_workspace/yuzijian): " WorkSpaceBind
while test -z "$WorkSpaceBind"
do
    read -p "输入内容为空，请重新输入: " WorkSpaceBind
done

# zerotier 网络 ID
read -p "请设置zerotier网络ID(例: 56374ac9a4148df5): " ZerotierNetID
while test -z "$ZerotierNetID"
do
    read -p "输入内容为空，请重新输入: " ZerotierNetID
done

# Launch container as root to init core Linux services and
# launch the Display Manager and greeter. Switches to
# unprivileged user after login.
# --device=/dev/tty0 makes session creation cleaner.
# --ipc=host is set to allow Xephyr to use SHM XImages
docker run -itd \
    --name $CONTAINER \
    --privileged \
    --restart=always \
    --device=/dev/tty0 \
    --device=/dev/net/tun \
    --cap-add=SYS_PTRACE \
    --gpus all \
    --shm-size 8g \
    --dns 8.8.8.8 --dns 8.8.4.4 \
    -p $NomachineBindPort:4000 \
    -p $SshBindPort:22 \
    -p $UdpBindPort:9993 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /sys/fs/cgroup:/sys/fs/cgroup \
    -v $WorkSpaceBind:/data:rw \
    $IMAGE

# initialize environment
echo "等待容器启动..."
sleep 5  # 等待容器完全启动
docker exec -it $CONTAINER bash /docker_config/set_env.sh $ZerotierNetID