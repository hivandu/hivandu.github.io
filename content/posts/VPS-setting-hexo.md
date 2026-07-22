---
title: "VPS 设置 Hexo"
date: 2014-11-02T00:00:00+08:00
draft: false
summary: "首先需要感谢@[lucifr](http://lucifr.com/)，我现在这篇文就是在 iPad 上登录 VPS 完成的。最后还是忍不住入手了下边两个 APP:"
---

![Prompt](https://cdn.jsdelivr.net/gh/hivandu/notes/img/202407291510190.png)

当然到现在并不完美，因为 rsync 和自动执行`generate`的代码我没有完成。安装[incrond](http://inotify.aiken.cz/?section=incron&page=download&lang=en)的时候总会出错，于是无法执行集群文件同步.所以现在还是在终端里执行`generate`和`cp -rf /home/xxxx/* /home/xxxx`

我这里并不是要教设置步骤，因为其实@lucifr 已经在他的[这篇文](http://lucifr.com/2013/06/02/hexo-on-cloud-with-dropbox-and-vps/)里写的很清楚了，我就写几点注意事项

1. 搞定 VPS 操作和基本的 Linux 命令很重要。
2. 要搞定 lnmp，参照这里的 lnmp 详细介绍
3. 新版本的 Hexo 有更改，在同一目录里是找不到/cli/generate.js 的，更别说 console.log 语句了
4. @lucifr 所说的新建立一个 Dropbox 账户，意思是在 VPS 主机上建立一个账户用来执行 Dropbox 同步，而不是新建立一个 Dropbox 账户。
5. 其他…

好吧，写其他是因为 iPad 上用 VI 进行编辑实在有点难受，现在先这样了，以后有时间了再写一个更详细的。

![snap_prompt_ssh](https://cdn.jsdelivr.net/gh/hivandu/notes/img/202407291511113.png)