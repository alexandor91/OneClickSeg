```bash
nohup sh docker_build.sh yuzijian 11.6.1 8 20.04 > ./docker_build_1.log 2>&1 &
sh docker_run.sh
```

TODO: 
- [ ] ssh key 的问题，不能直接传到 github
- [ ] 考虑添加显卡驱动，详见 GeForce-XorgDisplaySettingAuto.sh
- [ ] 考虑更换为 ubuntu desktop
