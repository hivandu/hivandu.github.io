---
title: "针对某一个项目自动切换 node 版本"
date: 2018-06-13T00:26:09+08:00
draft: false
summary: "`nvm`作为 node 的版本管理器，并不具备自动切换版本切换的功能，有的时候我们需要针对某一个项目切换当前的 node 版本，这个时候就需要用到其他工具了。比如`avn`

举例项目:`project`

因为最近 Node 更新到 10 之后，我将系统默认版本切换到了 10，有不更新不舒服斯基强迫症
而`project` 编译的版本为 8，否则会出现编译出错。


```shell
$ brew install nvm
$ nvm i -g avn
$ avn steup
```

之后在`project`根目录中添加一个文件`.node-version`"
---

```shell
$ touch .node-version
$ echo v8 >> .node-version #node 需要切换的版本
$ echo `source "$HOME/.avn/bin/avn.sh" # load avn` >> ~/.zshrc
```


这样就可以了。

不过不排除报错的情况，如果是`brew` 安装的`nvm`, 则默认`nvm.sh`并不在`~/.nvm`目录内，这个时候可能需要在执行一下某段脚本。一样添加到`~/.zshrc`内

```shell
$ echo `[[ -s "$(brew --prefix nvm)/nvm.sh" ]] && source $(brew --prefix nvm)/nvm.sh` >> ~/.zshrc
```

再切换一下项目目录

```shell
$ cd $project
$ avn activated v8.11.2 (avn-nvm v8.11.2)
```

至此完成了！