---
title: "在 Yosemite 中设置 Pow"
date: 2014-11-07T00:00:00+08:00
draft: false
summary: "在 Yosemite 中，Pow 安装和启动是有问题的。这是因为 ipfw 被移除了，所以如果要在 Yosemite 中跑 Pow，需要做些设置才可以。"
---

![image-20240729151301159](https://cdn.jsdelivr.net/gh/hivandu/notes/img/202407291527144.png)



1, 添加文件`com.pow`到`/etc/pf.anchors/`目录内

`sudo vim /etc/pf.anchors/com.pow`

2, 在文件内添加代码:

```
rdr pass on lo0 inet proto tcp from any to any port 80 -> 127.0.0.1 port 20559
rdr pass on en0 inet proto tcp from any to any port 80 -> 127.0.0.1 port 20559
rdr pass on en9 inet proto tcp from any to any port 80 -> 127.0.0.1 port 20559
```

**NOTE: 代码后一行必须要有换行符，否则会出现语法错误**

3, 打开文件`/etc/pf.conf`

4, 添加代码: `rdr-anchor "pow"`，需要添加到`rdr-anchor "com.apple/*"`下一行

5, 打开文件`/etc/pf.anchors/com.apple`, 并添加代码:

```
load anchor "pow" from "/etc/pf.anchors/com.pow"

```

**NOTE: 一样必须有换行符**

6, 终端执行:

`sudo pfctl -f /etc/pf.conf`

7, 好了，现在可以打开`pf`了:

`sudo pfctl -e`