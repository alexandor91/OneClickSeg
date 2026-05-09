#!/bin/bash
# 环境设置, 仅在初次启动时运行

# 更改 root 用户的密码
echo -n "请设置 root 用户密码: "
read -s PASSWORD
echo    # 保持这行，在密码输入后换行
echo "root:$PASSWORD" | chpasswd

# 添加 .bashrc 到末尾
echo "source /docker_config/.bashrc" >> /root/.bashrc

# Install zerotier with input net id
ZerotierNetID=$1
bash /docker_config/install_zerotier.sh $ZerotierNetID

# 设置用户自定义环境启动
# ...