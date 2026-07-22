---
title: "M1 安装 Homebrew(ARM)"
date: 2024-01-01T00:00:00+08:00
draft: false
---

?> 详情可见[作者说明](https://github.com/Homebrew/brew/issues/7857#issue-647960270)

## 安装

ARM 版本 Homebrew 必须安装在`/opt/homebrew`路径下

```shell
cd /opt
sudo mkdir homebrew
sudo curl -L https://github.com/Homebrew/brew/tarball/master | tar xz --strip 1 -C homebrew
```

如果不进行 sudo 授权，则会报错；



## 环境变量

本人使用`zsh`, 所以编辑文件`~/.zshrc`. 添加如下内容：

```shell
path=('/opt/homebrew/bin' $path) 
export PATH
```

?> 如果是使用`bash`，请修改`~/.bashrc`

在终端内执行:

```shell
source ~/.zshrc
```



现在可以试试执行`brew install graphviz`试试看能否正常安装回归树可视化模块；



## 软件包和迁徙

软件包依然需要使用 X86 版 Homebrew

```shell
arch -x86_64
```

启用一个 X86 模式中端，之后运行的命令都在 X86 模式下运行，再次安装 Homebrew

```shell
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```



!> 注意：要将 ARM 版本 Homebrew 环境变量设置到最前面，此时两个版本共存时会有限启动 ARM 版本，需要运行 X86 版本时，需要手动输入完整路径`arch -x86_64 /usr/local/bin/brew`



可以在配置文件中设置`alias`

```shell
abrew='/opt/homebrew/bin/brew' # ARM Homebrew
ibrew='arch -x86_64 /usr/local/bin/brew' # X86 Homebrew
```

如果对已有软件包做迁徙，则：

```
ibrew bundle dump
```

此时在目录下就得到一个名为`Brewfile`的备份文件，导入内容并安装

```shell
abrew bundle --file /path/to/Brewfile
```

!> 执行之前需要编辑`Brewfile`文件，将`cask`和`mas`开头的记录删除掉；





