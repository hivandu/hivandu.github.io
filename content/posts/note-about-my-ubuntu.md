---
title: "Note about my ubuntu"
date: 2024-01-01T00:00:00+08:00
draft: false
summary: "这是我为了便于自己以后方便而记录的一些关于我 ubuntu 上设置所需要的内容！当然其中内容都是通用的，只是如果你们要拿去用的话记得将路径以及一些变量变成你自己的！

### about system

`$ sudo gedit /etc/apt/sources.list`"
---

```
deb http://mirrors.163.com/ubuntu/ karmic main restricted universe multiverse
deb http://mirrors.163.com/ubuntu/ karmic-security main restricted universe multiverse
deb http://mirrors.163.com/ubuntu/ karmic-updates main restricted universe multiverse
deb http://mirrors.163.com/ubuntu/ karmic-proposed main restricted universe multiverse
deb http://mirrors.163.com/ubuntu/ karmic-backports main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ karmic main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ karmic-security main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ karmic-updates main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ karmic-proposed main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ karmic-bac.     kports main restricted universe multiverse
```

`$ sudo apt-get update and $ sudo apt-get upgrade`

### and hosts

`$ gedit /etc/hosts # My Dropbox have new hosts file`

#### about ATI

Donload form [This link](http://support.amd.com/cn/Pages/AMDSupportHub.aspx)

```
cd **
sudo sh **.run
```

That's ok

### about Keepass 2

so...apt-get

```
$ sudo apt-add-repository ppa:jtaylor/keepass
$ sudo apt-get update
$ sudo apt-get install keepass2
```

10.04 所需[软件包](http://packages.debian.org/sid/keepass2)

`1 install mono #[link](http://mono-project.com/DistroPackages/Ubuntu)`

```
Click on "System", "Administration", "Software Sources".
Click on the "Other Software" tab.
Click on "Add...", and enter the line:**deb http://badgerports.org lucid main**
Click on "Add Source"
Click on "Authentication", then on "Import Key File"
Download this [GPG key file](http://badgerports.org/directhex.ppa.asc), ID 0E1FAD0C, and select it in the "Import Key File" window
Click on "Close", then "Reload" when the pop-up appears. You're all set!
```

`2 $ sudo apt-add-repository ppa:jtaylor/keepass`

`3 $ sudo apt-get update`

`4 $ sudo apt-get install keepass2`

### about Java

Download from [This link](http://jdk7.java.net/download.html)

and

```
$ cd **
$ tar **.tar.gz(64bit)
$ sudo update-alternatives --install "/usr/bin/java" "java" "/home/hivan/software/jdk1.7.0_04/bin/java" 1
$ sudo update-alternatives --config java
```

**其实就是配置一下默认路径就 OK 了！**

或者以下方法：

`$ gedit ~/.bashrc`

末尾添加变量

```
JAVA_HOME=/home/hivan/software/jdk1.7.0_04
JRE_HOME=${JAVA_HOME}/jre
CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib
PATH=${JAVA_HOME}/bin:$PATH
```

and

```
$ source ~/.bashrc
$ sudo update-alternatives --install /home/hivan/software/jdk1.7.0_04/bin/java 300  
$ sudo update-alternatives --install /home/hivan/software/jdk1.7.0_04/bin/javac 300  
$ sudo update-alternatives --install /home/hivan/software/jdk1.7.0_04/bin/jar 300   
$ sudo update-alternatives --config java  #用于替换 Java，可能会跳出选择框
$ java -version #测试
java version "1.7.0_04-ea"
Java(TM) SE Runtime Environment (build 1.7.0_04-ea-b19)
Java HotSpot(TM) 64-Bit Server VM (build 23.0-b20, mixed mode)
```

### about goagent

```
$ cd
$ mkdir bin
$ cd bin
建立一个 proxy.sh file
$ gedit proxy.sh
输入： python /***/goagent/local/proxy.py save and quit
$ chmod 700 proxy.sh
```

### about ruby

```
$ sudo apt-get install apache2 curl git libmysqlclient-dev mysql-server nodejs
$ bash -s stable < <(curl -s https://raw.github.com/wayneeseguin/rvm/master/binscripts/rvm-installer)
$ echo '[[ -s "$HOME/.rvm/scripts/rvm" ]] && . "$HOME/.rvm/scripts/rvm" # Load RVM function' >> ~/.bashrc
$ source .bashrc
$ rvm requirements
$ sudo apt-get install build-essential openssl libreadline6 libreadline6-dev curl git-core zlib1g zlib1g-dev libssl-dev libyaml-dev libsqlite3-0 libsqlite3-dev sqlite3 libxml2-dev libxslt-dev autoconf libc6-dev ncurses-dev automake libtool bison subversion
$ rvm install 1.9.3
$ rvm 1.9.3 –-default #有可能出错**RVM is not a function, selecting rubies with 'rvm use ...' will not work.** 则执行：`$ rvm alias create default 1.9.3
```

### about Rails

```
$ gem install bundler rails rdoc #rdoc 为 Octopress 所需组建
$ rails -v #如果“**程序“rvm”尚未安装。**”则检查一下.bashrc 的路径配置
```

### about python

```
$ python -V
Python 2.6.6
$ curl -kL http://github.com/utahta/pythonbrew/raw/master/pythonbrew-install | bash
$ . $HOME/.pythonbrew/etc/bashrc
$ pythonbrew install 2.7.1
$ pythonbrew switch 2.7.1
Switched to Python-2.7.1
$ python -V
Python 2.7.1
```

ubuntu 12 中默认是 2.7，如果要安装 3.2 和上述步骤一样！改一下版本号

### about Sublime Text 2

```
#!/usr/bin/env xdg-open
[Desktop Entry]
Name=Sublime Text 2
Comment=Sublime Text 2
Exec=/home/hivan/software/"Sublime Text 2"/sublime_text
Icon=/home/hivan/software/Sublime Text 2/Icon/128X128/sublime_text.png
Terminal=false
Type=Application
Categories=Application;Development;
StartupNotify=true
```

**设置权限：可执行文件！**

and

```
$ cd /home/hivan/software/"Sublime Text 2"/
$ sudo cp "Sublime Text 2.desktop" /usr/share/applications
```

Ctrl+`

```
import urllib2,os;pf='Package Control.sublime-package';ipp=sublime.installed_packages_path();os.makedirs(ipp) if not os.path.exists(ipp) else None;open(os.path.join(ipp,pf),'wb').write(urllib2.urlopen('http://sublime.wbond.net/'+pf.replace(' ','%20')).read())
```

Ctrl+Shift+P

1 ZenCoding

2 Alignment

3 Markdown

4 setting user

{
        "ignored_packages": []
}

download setting from [this link](https://github.com/hivan/SomeSetting/blob/master/SubT2Setting.md)

Finally, input

right, you can only choose scim,so...

```
$ sudo apt-get install scim
$ sudo apt-get install scim-pinyin
```

...

```
scim 设置－>全局设置－>将预编辑字符串嵌入到客户端中勾去掉
scim 设置->gtk->嵌入式候选词标 勾去掉
```

### about Retext

```
sudo add-apt-repository ppa:mitya57
sudo apt-get update
sudo apt-get install retext
```

or [This link](http://sourceforge.net/p/retext/blog/2012/03/retext-300-released/)

### about Octopress

Go to [This Link](http://hivan.me/octopress-install-to-windows8/)

add and remove ppa

```
$ sudo add-apt-repository ppa:name/name
$ sudo add-apt-repository -r ppa:name/name
```

Temporary end, to be continued...