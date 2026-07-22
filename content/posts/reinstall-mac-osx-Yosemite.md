---
title: "重裝「Yosemite」"
date: 2024-01-01T00:00:00+08:00
draft: false
---

重裝系統之後，很多東西需要重裝，特別是開發環境。而開發環境的先後順序和設置，一直是我頭疼的事情。這次就着從新安裝了一遍，把很多東西都記錄下來。之前那個環境被我搞的亂七八糟，並且恢復不回來了。

![image][1]

<!--more-->

多大多數次序參照 @[mrzhang][2]

## 系統偏好設置

- 更改電腦名稱

  共享

- 允許安裝任何來源 APP

  安全性與隱私 --》通用

- 設置快捷鍵

  鍵盤 --》 快捷鍵

## 配置 VPN 以及 SSH

  **很重要，因爲很多源都在牆外了**

## 安裝輸入法
- 下載並安裝"[Squirrel][3]"
- 下載並安裝"SCU" 

安裝[Sublime Text 3][4]

[設置 package][5]

`Ctrl+ ,` and [setting][6]:

```
{
  "caret\_style": "phase",
  "color\_scheme": "Packages/Color Scheme - Default/Solarized (Light).tmTheme",
  "font\_face": "Monaco",
  "font\_size": 13.0,
  "hightlight\_line": true,
  "hightlight\_modified\_tabs": true,
  "ignored\_packages":
  [
	"Vintage"
  ],
  "indent\_to\_bracket": true,
  "draw\_centered": false, //居中显示
  "line\_numbers": true, //显示行号
  "gutter": true, //显示行号边栏
  "fold\_buttons": true, //显示折叠按钮
  "fade\_fold\_buttons": true, //始终显示折叠按钮
  "rulers": [], //列显示垂直标尺，在中括号里填写数字，宽度按字符计算
  "spell\_check": false, //拼写检查
  "hot\_exit": true, //保留未保存内容
  "line\_padding\_bottom": 1,
  "line\_padding\_top": 1,
  "scroll\_past\_end": true, //文本最下方缓冲区
  "tab\_size": 2, // Tab 制表宽度
  "translate\_tabs\_to\_spaces": true, //缩进和遇到 Tab 键用空格替代
  "wide\_caret": true,
  "word\_wrap": true,
  "match\_tags": true, //HTML 下突出显示光标所在标签的两端。
  "match\_selection": true, //全文高亮当前选中字符
  "wrap\_width": 80
}
```

## 編輯設置 /etc/paths

```
/usr/local/bin
/usr/local/sbin
/usr/bin
/usr/sbin
/bin
/sbin
```

## 安裝[Xcode][7]

```
xcode-select --install
```

## 安裝[Homebrew][8]

```
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/homebrew/go/install)"
```


**PS: 這裏可能會很長時間的等待**

## 設置 Sublime 終端鏈接

```
ln -s /Applications/Sublime\ Text.app/Contents/SharedSupport/bin/subl /usr/local/bin/sm
```

## Git, [autojump][9]

```
brew install git autojump
```

## [Oh My Zsh][10]

```
curl https://raw.githubusercontent.com/creationix/nvm/master/install.sh | sh
```

設置 `~/.zshrc`:

```
export NVM\_NODEJS\_ORG\_MIRROR="http://npm.taobao.org/dist"

[[ -s "$HOME/.nvm/nvm.sh" ]] && . "$HOME/.nvm/nvm.sh"
export NODE\_PATH=$NVM\_DIR/$(nvm\_ls current)/lib/node\_modules
```

## 安裝 NodeJS

```
nvm install 0.11.15
nvm alias default 0.11.15
```

## 安裝[rbenv][11]

```
git clone git://github.com/sstephenson/rbenv.git \~/.rbenv
# 用来编译安装 ruby
git clone git://github.com/sstephenson/ruby-build.git \~/.rbenv/plugins/ruby-build
# 用来管理 gemset, 可选, 因为有 bundler 也没什么必要
git clone git://github.com/jamis/rbenv-gemset.git  \~/.rbenv/plugins/rbenv-gemset
# 通过 gem 命令安装完 gem 后无需手动输入 rbenv rehash 命令, 推荐
git clone git://github.com/sstephenson/rbenv-gem-rehash.git \~/.rbenv/plugins/rbenv-gem-rehash
# 通过 rbenv update 命令来更新 rbenv 以及所有插件, 推荐
git clone https://github.com/rkh/rbenv-update.git \~/.rbenv/plugins/rbenv-update
```

#### 設置`~/.zshrc`

```
export PATH="$HOME/.rbenv/bin:$PATH"
eval "$(rbenv init -)"
```

#### [其他][12]

## 安裝 Ruby

```
rbenv install -l      # list all available versions
rbenv install 2.1.5   # install a Ruby version
rbenv global 2.1.5    # set the global version
rbenv versions        # list all installed Ruby versions
```

## 配置 gem 源

```
gem sources -a http://ruby.taobao.org/ -r https://rubygems.org/
echo 'gem: --no-document' \>\> \~/.gemrc
gem update
gem update --system
```

## 安装 MongoDB, MySQL

```
brew install mongodb mysql
```

設置開機自啓動「可選」

```
mkdir -p \~/Library/LaunchAgents
ln -sfv /usr/local/opt/mongodb/\*.plist \~/Library/LaunchAgents
ln -sfv /usr/local/opt/mysql/\*.plist \~/Library/LaunchAgents
```

## 安装 [Pow][13]

```
curl get.pow.cx | sh
gem install powder
```

[Powder][14] 是一套管理工具

## SSH-KeyGen

```
ssh-keygen -t rsa
cat \~/.ssh/id\_rsa.pub
```

## 安裝 Rails, sass, compass 以及 hexo

```
gem install rails sass compass
npm install -g hexo
```

#### 安裝必要工具

```
gem install mysql2
gem install capistrano
gem install capistrano-ext

```

## Snippets -  [Download](Snippets - Download)

## 安裝其他 APP

[1]:	https://qiniu.hivan.me/Yosemite.jpg
[2]:	http://mrzhang.me/blog/after-reinstall-the-system.html
[3]:	https://code.google.com/p/rimeime/
[4]:	http://www.sublimetext.com/3
[5]:	https://packagecontrol.io/installation
[6]:	https://github.com/hivan/SomeSetting/blob/master/SubT2Setting.md
[7]:	https://developer.apple.com/xcode/
[8]:	http://brew.sh/
[9]:	https://github.com/joelthelion/autojump
[10]:	http://ohmyz.sh/
[11]:	https://github.com/sstephenson/rbenv
[12]:	https://ruby-china.org/wiki/rbenv-guide
[13]:	http://pow.cx/
[14]:	https://github.com/Rodreegez/powder