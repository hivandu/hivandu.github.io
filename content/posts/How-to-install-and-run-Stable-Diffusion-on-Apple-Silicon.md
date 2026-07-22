---
title: "在 Apple Silicon M1/M2 Mac 上安装和运行 Stable Diffusion"
date: 2023-02-23T16:10:03+08:00
draft: false
tags: []
summary: "说实话，我找了好多关于如何在 M1/M2 上安装和运行 Stable Diffusion 的教程和帖子，发现相互之间借鉴的不少，但是能用的确实没几个。

寻找一番后，发现其实没那么复杂。也不知道为什么网上的那么多教程搞得那么复杂，又是这个又是那个的一大堆，简单实现的方式有好几种："
---

## 1. Diffuers

这是可以在 App Store 上直接搜索并下载的一个 App，看评分和排名似乎都不太好，开发者却是「Hugging Face」，其实在官方的 Github 上就有其下载链接：‣

 

![notion image](https://qiniu.hivan.me/picGo/20230601161228.png?imgNote)

 

这应该是体验 Stable Diffusion 最简便的方式了吧。而且还支持选择 Model，

![notion image](https://qiniu.hivan.me/picGo/20230601161512.png?imgNote)

不过有点遗憾的点是没办法调整参数。

 

![notion image](https://qiniu.hivan.me/picGo/20230601161505.png?imgNote)

 

## 2. DiffusionBee

这是出现的比较早的一款第三方 App，使用起来也是特别简单，直接下载安装就行了：

DiffusionBee - Stable Diffusion App for AI Art

DiffusionBee is the easiest way to generate AI art on your computer with Stable Diffusion. Completely free of charge.

![DiffusionBee - Stable Diffusion App for AI Art](https://qiniu.hivan.me/picGo/20230601161520.png?imgNote)

https://diffusionbee.com/download

![DiffusionBee - Stable Diffusion App for AI Art](https://qiniu.hivan.me/picGo/20230601161523.jpg?imgNote)

目前不止是 MacOS，还有对应 Windows 64 Bit 的版本，而且，你可以选择下载 HQ Version 版本。官方对其说的是速度慢两倍，但是图像质量更好。

![notion image](https://qiniu.hivan.me/picGo/20230601161530.png?imgNote)

以上两个 App 第一次使用的时候都是需要下载 Model 的，之后就可以直接开心的玩耍了，相比较而言，DiffusionBee 在参数选择上要多一点。支持 Text to Image, Image To Image 等。

![notion image](https://qiniu.hivan.me/picGo/20230601161538.png?imgNote)

## # 安装 AUTOMATIC1111

这个方法是需要有一点动手能力了，不过相比较而言，也不是那些网站上介绍的那么繁琐。其实就只需要几步而已。

 

### 1. 下载 Homebrew

一个包管理器，不太明白的朋友不需要管那些，操作就行了

打开你的终端，不明白什么是终端直接在你的搜索框里输入”终端”，或者”Terminal”, 就能看到了。

![notion image](https://qiniu.hivan.me/picGo/20230601161546.png?imgNote)

然后直接把下面的代码粘贴进去，回车，看着他跑就行了

```shell
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

 

### 2. 安装一些必须的软件包

等上面步骤跑完之后，再复制下面的代码，一样粘贴进去回车看着他跑：

```shell
brew install cmake protobuf rust python@3.10 git wget
```

![notion image](https://qiniu.hivan.me/picGo/20230601161559.png?imgNote)

### 3. 下载 AUTOMATIC1111 存储库

等上面步骤跑完，在你的终端输入一下代码并回车

```shell
cd ~
```

以上命令是为了让你进入你在 Mac 电脑上的账户主目录，就是这个地址

```shell
/User/xx/
```

然后我们接下来的操作会在你这个目录下下载一个文件夹，名字叫 「stable-diffusion-webui」，这贴下面代码到你的终端里，然后回车

```shell
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
```

命令跑完后，你就会渐渐自己名字的目录下多了一个 stable-diffusion-webui 的目录了，然后在终端进入这个目录，操作方法和上面一样

```shell
cd ~/stable-diffusion-webui/
```

![notion image](https://qiniu.hivan.me/picGo/20230601161606.png?imgNote)

在你的访达里你也进入这个目录，然后继续进入 models/Stable-diffusion， 这里是存放 Model 的地方。现在你需要下载一个 Model 存放进去，你可以直接在这里下载 [1.5 model](https://stable-diffusion-art.com/models/#Stable_diffusion_v15), 当然，如果你需要 2.1 的或者其他 model，可以去点击下面的链接进去自己下载一个合适的，然后扔到目录里。

Models - Hugging Face

We’re on a journey to advance and democratize artificial intelligence through open source and open science.

![Models - Hugging Face](https://qiniu.hivan.me/picGo/20230601161611.ico?imgNote)

https://huggingface.co/models?sort=downloads&search=stable+diffusion

![Models - Hugging Face](https://qiniu.hivan.me/picGo/20230601161614.png?imgNote)

然后你的目录应该会是这样：

![notion image](https://qiniu.hivan.me/picGo/20230601161619.png?imgNote)

然后你就可以尝试着跑你的 stable diffusion 了，刚才我们在终端里进入了 ~/stable-diffusion-webui/， 假设你还在这个位置，我们就可以直接输入：

```shell
./webui.sh
```

![notion image](https://qiniu.hivan.me/picGo/20230601161624.png?imgNote)

然后去干些自己的事情吧，喝杯茶，看看书。要跑一会呢，特别是你网络不好的情况。

等到终端命令全部跑完后，打开你的 Safari，输入：http://127.0.0.1:7860/

好了，可以把玩了。

![notion image](https://qiniu.hivan.me/picGo/20230601161631.png?imgNote)

 

## 4. 其他

是的，还没结束，还有一些要说的，其实在 Mac App Store 里搜索的话，你还能看到一些其他的 App 可以直接使用，比如：

![notion image](https://qiniu.hivan.me/picGo/20230601161638.png?imgNote)

反正都是免费的，尽量多试试，找到一个自己满意的。

另外，不管你用那种方法，你都需要知道一些 good prompts for Stable Diffusion, 这里有一个地方可以看些别人的例子，不过不是那么容易打开：

arthub.ai

![arthub.ai](https://qiniu.hivan.me/picGo/20230601161647.ico?imgNote)

https://arthub.ai/

还有啊，自己可以多测试一些 model，下下来把玩下。



祝你玩的愉快。全文完。