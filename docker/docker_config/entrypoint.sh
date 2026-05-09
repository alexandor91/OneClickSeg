#!/bin/sh
# 启动 X 虚拟帧缓冲
Xvfb :1 -screen 0 1024x768x16 &
export DISPLAY=:1
# start ssh & sshd & dbus & nomachine & zerotier
service ssh start
/usr/sbin/sshd
/etc/init.d/dbus start
/etc/NX/nxserver --startup
tail -f /usr/NX/var/log/nxserver.log &

# 启动ZeroTier服务
if [ -f "/var/lib/zerotier-one/zerotier-one" ]; then
    echo "启动ZeroTier服务..."
    /etc/init.d/zerotier-one start || /var/lib/zerotier-one/zerotier-one -d
    
    # 加载之前保存的iptables规则
    if [ -f "/etc/iptables/rules.v4" ]; then
        echo "正在恢复之前保存的iptables规则..."
        iptables-restore < /etc/iptables/rules.v4
        echo "iptables规则已恢复"
    fi
fi

# close gnome animations
sudo gsettings set org.gnome.desktop.interface enable-animations false
# custom startup for user
if [ -f "/docker_config/custom_startup.sh" ]; then
	bash /docker_config/custom_startup.sh
fi
# Start interactive bash shell
/bin/bash