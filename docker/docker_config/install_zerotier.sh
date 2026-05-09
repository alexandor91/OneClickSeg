#!/bin/bash
# Install zerotier, change it with your own net id
curl -s https://install.zerotier.com | bash

# 加入zerotier网络
net_id=$1
if [ -z "$net_id" ]; then
    echo "请提供ZeroTier网络ID作为参数"
    exit 1
fi

# 等待ZeroTier服务启动
sleep 5
/etc/init.d/zerotier-one start
sleep 2

# 加入zerotier网络
zerotier-cli join $net_id

# 配置使用 zerotier 路由转发
zerotier-cli set $net_id allowGlobal=1
zerotier-cli set $net_id allowDefault=1

# 配置 /etc/sysctl.conf 文件
echo "net.ipv4.ip_forward = 1" >> /etc/sysctl.conf
echo "fs.inotify.max_user_instances=512" >> /etc/sysctl.conf
echo "fs.inotify.max_user_watches=262144" >> /etc/sysctl.conf
sysctl -p /etc/sysctl.conf

# 确保 iptables 包已安装
apt-get update && apt-get install -y iptables iproute2 && rm -rf /var/lib/apt/lists/*

# 获取网卡名称, PHY_IFACE=连接内网的网卡的名字, ZT_IFACE=zerotier的虚拟网卡名字
PHY_IFACE=$(ip -4 route list 0/0 | sort -k 9 -n | head -1 | awk '{print $5}')
# 等待ZeroTier网络建立
echo "等待ZeroTier网络建立..."
timeout=30
while [ $timeout -gt 0 ]; do
    ZT_IFACE=$(zerotier-cli listnetworks | grep $net_id | awk '{print $7}')
    if [ ! -z "$ZT_IFACE" ]; then
        break
    fi
    sleep 2
    timeout=$((timeout-2))
done

if [ -z "$ZT_IFACE" ]; then
    echo "ZeroTier网络接口未就绪，请稍后再试"
    exit 1
fi

echo "物理网卡: $PHY_IFACE, ZeroTier网卡: $ZT_IFACE"

# 配置 IPv4 路由转发
iptables -t nat -A POSTROUTING -o $PHY_IFACE -j MASQUERADE
iptables -A FORWARD -i $PHY_IFACE -o $ZT_IFACE -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
iptables -A FORWARD -i $ZT_IFACE -o $PHY_IFACE -j ACCEPT

# 保存路由表
mkdir -p /etc/iptables
iptables-save > /etc/iptables/rules.v4
echo "iptables规则已保存"

echo "ZeroTier配置完成，网络ID: $net_id"
zerotier-cli info
zerotier-cli listnetworks