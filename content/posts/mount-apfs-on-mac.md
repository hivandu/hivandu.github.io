---
title: "Mac 上挂载 APFS 移动硬盘"
date: 2023-10-26T18:10:41+08:00
draft: false
tags: []
summary: "> 自用文，有需要的自取。

百度网盘同步会认为移动硬盘是系统盘，所以无法进行同步。当然，也有例外的，之前我也是不知怎么同步的。

这次设置的时候被警告了，不允许设置。"
---

好吧，那就只能将移动硬盘挂载到我的用户目录里了，我的移动硬盘是 APFS 类型，执行下面命令：

```
# 查看当前硬盘 IDENTIFIER
diskutil apfs list

# 或者下面这段命令
diskutil list

# 然后需要进行解锁，恢复键值(recovery_key):
diskutil apfs unlockVolume /dev/apfs_volume_id -passphrase recovery_key

# 接着进行装载到自己期望的目录
diskutil mount -mountPoint Path apfs_volume_id
```

假定我的硬盘`apfs_volume_id`为 disk5s1, 希望挂载到`~/mount`则：

```shell
diskutil apfs unlockVolume /dev/disk5s1 -passphrase recovery_key
diskutil mount -mountPoint ~/mount /dev/disk5s1
```

本是留待自用的，有需要的有缘人自行取走。