---
title: "Apple M1 的 AI 环境搭建"
date: 2021-10-21T00:00:00+08:00
draft: false
summary: "> 本文环境搭建的基础是 Python3.9， 因为 M1 为 ARM 架构，所以放弃了 Anaconda，使用 Miniforge3。包括 Tensorflow, xgboost, Lightgbm, Numpy, Pandas, Matplotlib, NGBoost 等。当然，因为是 Python3.9， 所以有些库实在是无法使用。"
---

## Homebrew

作为 Mac 的包管理神器，首先当然要先从 Homebrew 开始。Homebrew 已经支持了 ARM 架构，可以直接进行安装，当然，如果你电脑里以前存在 X86 的 brew 支持，请先卸载干净。

### Homebrew 卸载

```bash
/bin/bash -c "$(curl -fsSL https://cdn.jsdelivr.net/gh/ineo6/homebrew-install/uninstall.sh)"
```

### Install ARM Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://cdn.jsdelivr.net/gh/ineo6/homebrew-install/install.sh)"
```

执行完毕后，Homebrew 安装在`/opt/homebrew`路径下；在安装完毕后，命令行后会提示执行命令设置环境变量，当然，以防万一，这里也提供一下：

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

如果是 bash shell， 则：

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.bash_profile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

记得`source ~/.zprofile`

### Install X86 Homebrew

```bash
arch -x86_64 /bin/bash -c "$(curl -fsSL https://cdn.jsdelivr.net/gh/ineo6/homebrew-install/install.sh)"
```

X86 版本的安装执行完成后命令行未提示添加环境变量。

### alias 支持多版本

在终端执行：

```bash
alias brew='arch -arm64 /opt/homebrew/bin/brew'
alias ibrew='arch -x86_64 /usr/local/bin/brew'
```
> 这里可以看出两者路径区别

### 设置镜像



#### 中科大源

```bash
# brew
git -C "$(brew --repo)" remote set-url origin https://mirrors.ustc.edu.cn/brew.git

# core
git -C "$(brew --repo homebrew/core)" remote set-url origin https://mirrors.ustc.edu.cn/homebrew-core.git

# cask
git -C "$(brew --repo homebrew/cask)" remote set-url origin https://mirrors.ustc.edu.cn/homebrew-cask.git

brew update

```



#### 清华大学源

```bash
# brew
git -C "$(brew --repo)" remote set-url origin https://mirrors.tuna.tsinghua.edu.cn/git/homebrew/brew.git

# core
git -C "$(brew --repo homebrew/core)" remote set-url origin https://mirrors.tuna.tsinghua.edu.cn/git/homebrew/homebrew-core.git

# cask
git -C "$(brew --repo homebrew/cask)" remote set-url origin https://mirrors.tuna.tsinghua.edu.cn/git/homebrew/homebrew-cask.git

brew update

```



#### 恢复默认源

```bash
# brew
git -C "$(brew --repo)" remote set-url origin https://github.com/Homebrew/brew.git

# core
git -C "$(brew --repo homebrew/core)" remote set-url origin https://github.com/Homebrew/homebrew-core.git

# cask
git -C "$(brew --repo homebrew/cask)" remote set-url origin https://github.com/Homebrew/homebrew-cask.git

brew update

```



[更多源](https://brew.idayer.com/guide/change-source/)



### Homebrew 其他相关



#### 设置 bottles 镜像


```bash
# bottles for zsh
echo 'export HOMEBREW_BOTTLE_DOMAIN=https://mirrors.ustc.edu.cn/homebrew-bottles/bottles' >> ~/.zprofile
source ~/.zprofile

# bottles bash
echo 'export HOMEBREW_BOTTLE_DOMAIN=https://mirrors.ustc.edu.cn/homebrew-bottles/bottles' >> ~/.bash_profile
source ~/.bash_profile
```



#### cask

目前 cask 是从 GitHub 上读取软件源，而 GitHub Api 对访问有限制，如果使用比较频繁的话，可以申请 Api Token，然后在环境变量中配置到`HOMEBREW_GITHUB_API_TOKEN`。

```bash
echo 'export HOMEBREW_GITHUB_API_TOKEN=yourtoken' >> ~/.zprofile
source ~/.zprofile
```



## Install Miniforge3


首先需要下载安装包： [Download](https://github.com/conda-forge/miniforge)

请下载 arm64(Apple Silicon)版本：

![image-20210908234235884](https://qiniu.hivan.me/picGo/20210908234236.png?imgNote)

下载完成后进入到文件目录，比如我是在`~/Download/`内，执行：

```bash
bash Miniforge3-MacOSX-arm64.sh
```

整个执行过程会有大概三次填写`yes`并回车确定，最后一次会询问你是否执行`conda init`， 会自动在`~/.zshrc`内添加环境变量，如果未执行的，可以将下面语句加入文件末尾：

```bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/Users/xx/miniforge3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/Users/xx/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/Users/xx/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/Users/xx/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate tf
# <<< conda initialize <<<
```

> 记得自行更改`/Users/xx/`内的用户名

等待 Miniforge3 安装完成，然后设置一个专供学习 Tensorflow 的虚拟环境

```bash
conda create -n tf python=3.9.5
conda activate tf # 将这句添加到~/.zshrc 内，每次打开 shell 都会自动执行
```

> 关于 conda 切换环境的命令，建议自行 Google 学习一下，很有用。

## Install Tensorflow

目前网上流传的 Tensorflow 安装基本是两个版本，一个是安装一大堆的支持和依赖，一个是使用`yml`文件提前准备好环境库一键完成环境创建，比如`environment.yml`：

```bash
conda env create --file=environment.yml --name=tf
```

其实这一步也很简单，Apple 为了大力推广自家的 ARM，已经为大家做好了这部分准备，我们只需要安装就行了。

> 假设目前在`tf`环境内

```bash
conda install -c apple tensorflow-deps
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal
```

好了，结束！

可以自行利用下面一段代码测试下：

```python
from tensorflow.keras import layers
from tensorflow.keras import models
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
```

![image-20210909000423413](https://qiniu.hivan.me/picGo/20210909000423.png?imgNote)

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)
test_loss, test_acc = model.evaluate(test_images, test_labels)
test_acc
```

![image-20210908235926168](https://qiniu.hivan.me/picGo/20210908235926.png?imgNote)

执行过程中可以在资源管理器中看到 GPU 的占用：

![image-20210909000541931](https://qiniu.hivan.me/picGo/20210909000541.png?imgNote)

## 其他

### Lightgbm

```bash
conda install Loghtgbm
```

一句代码解决，完全靠谱。

### xgboost

xgboost 稍微有点麻烦，我测试了最稳妥的安装方式，还是自行编译，那这个时候我们就需要用到`brew`安装并设置编译环境了：

> 注意，我用的都是`brew`而非`ibrew`, 目前都是在 ARM 环境下完成操作。

```bash
brew install gcc
brew install cmake
brew install libomp
```

然后下载源码并执行

```bash
git clone git@github.com:dmlc/xgboost.git
cd xgboost
mkdir build
cd build
CC=gcc-11 CXX=g++-11 cmake ..
cd ../python-package
/Users/xx/miniforge3/envs/tf/bin/python setup.py install
```

然后就 OK 了。

至于其他的，Numpy 在安装 Tensorflow 的时候就自动作为依赖安装了，Pandas, Matplotlib, NGBoost 等，执行下方：

```bash
conda install -c conda-forge pandas
conda install -c conda-forge matplotlib
conda install -c conda-forge ngboost
```

如果 conda 内实在没有的，再试试 pip 安装，再不行，就只能自行下载源码编译了。

目前在当前环境下解决不了的几个库：

1. [CatBoost](https://catboost.ai/)
2. [Cairo](https://www.cairographics.org/) -> [Pycairo](https://pycairo.readthedocs.io/en/latest/)
3. [GraphEmbedding](https://github.com/shenweichen/GraphEmbedding)
4. [CV2](https://opencv.org/)
5. [igraph](https://igraph.org/)

> 在整个过程中，可能会遇到各种各样的问题，大家要习惯于使用 Google 和查阅官方文档；



## 参考

[Tensoflow-macos](https://github.com/apple/tensorflow_macos)

[Run xgboost on Mac and Regression data](https://notes.hivan.me/#/AI_Data/run_xgboost_on_M1_and_regression)

[Accelerating TensorFlow Performance on Mac](https://blog.tensorflow.org/2020/11/accelerating-tensorflow-performance-on-mac.html)

[The new Apple M1 chips have accelerated TensorFlow support](https://www.reddit.com/r/MachineLearning/comments/js2p7s/n_the_new_apple_m1_chips_have_accelerated/)

[M1 Mac Mini Scores Higher Than My RTX 2080Ti in TensorFlow Speed Test.](https://medium.com/analytics-vidhya/m1-mac-mini-scores-higher-than-my-nvidia-rtx-2080ti-in-tensorflow-speed-test-9f3db2b02d74)

[GPU acceleration for Apple's M1 chip?](https://github.com/pytorch/pytorch/issues/47702)

[M1 芯片 Mac 上 Homebrew 安装教程](https://zhuanlan.zhihu.com/p/341831809)

[Mac mini M1 使用简单体验(编程、游戏、深度学习)](https://zhuanlan.zhihu.com/p/343195119)

[Installing TensorFlow 2.4 on MacOS 11.0 without CUDA for both Intel and M1 based Macs](https://medium.datadriveninvestor.com/installing-tensorflow-2-4-on-macos-11-0-without-cuda-for-both-intel-and-m1-based-macs-a1c4edf1dbab)

[在 M1 芯片 Mac 上使用 Homebrew](https://sspai.com/post/63935)

[Apple M1 终于让 MacBook 变的可以炼丹了](https://zhuanlan.zhihu.com/p/304488647)

[Install XGBoost and LightGBM on Apple M1 Macs](https://towardsdatascience.com/install-xgboost-and-lightgbm-on-apple-m1-macs-cb75180a2dda)

[Installing TensorFlow on the M1 Mac](https://towardsdatascience.com/installing-tensorflow-on-the-m1-mac-410bb36b776)

[Getting Started with tensorflow-metal PluggableDevice](https://developer.apple.com/metal/tensorflow-plugin/)

[M1 芯片 mac 安装 xgboost 和 lightgbm](https://blog.csdn.net/weixin_41411460/article/details/112358379)

[AI - Apple Silicon Mac M1 机器学习环境 (TensorFlow, JupyterLab, VSCode)](https://makeoptim.com/deep-learning/mac-m1-tensorflow)

[M1 芯片安装 tensorflow](https://www.codeleading.com/article/36895460744/)

[使用 MacBook pro M1 搭建基于 ML Compute 加速的 TensorFlow 深度学习环境](https://zhuanlan.zhihu.com/p/343769603)

[你的 Mac 有了专用版 TensorFlow，GPU 可用于训练，速度最高提升 7 倍](https://www.jiqizhixin.com/articles/2020-11-19-3)

[在 M1 的 Mac 上安装 Tensorflow（避坑版）](https://blog.csdn.net/hsywatchingu/article/details/118055508)

[在 M1 芯片 Mac 上搭建原生适配 Python 环境](https://zhuanlan.zhihu.com/p/368680708)

[Conda-forge Miniforge](https://github.com/conda-forge/miniforge)

[M1 mac 安装 PyTorch 的完整步骤指南](https://zhuanlan.zhihu.com/p/394514049)

[macOS M1(AppleSilicon) 安装 TensorFlow 环境](https://zhuanlan.zhihu.com/p/349409718)

[傻瓜版 M1 配置 Tensorflow-超简单近乎一键完成](https://zhuanlan.zhihu.com/p/367512283)

[environment.yml](https://raw.githubusercontent.com/mwidjaja1/DSOnMacARM/main/environment.yml)

[opencv-python](https://pypi.org/project/opencv-python/)

[MAC 安装 Opencv 以及 Dlib 碰到的一些问题](https://zhuanlan.zhihu.com/p/28448206)

[Jupiter Widgets](https://ipywidgets.readthedocs.io/en/stable/user_install.html)

[启动 SparkContext 报错](https://blog.csdn.net/Jarry_cm/article/details/106069025)

[MacBook Pro 2020 M1 芯片安装 xgboost](https://zhuanlan.zhihu.com/p/351496785)

[xgboost](https://xgboost.readthedocs.io/en/latest/install.html#installation-guide)

[Homebrew / Linuxbrew 镜像使用帮助](https://mirrors.tuna.tsinghua.edu.cn/help/homebrew/)

[镜像助手](https://brew.idayer.com/guide/change-source/)

[Apple Silicon Mac 安装 xgboost](https://zhuanlan.zhihu.com/p/357060236)

[M1 芯片 mac 安装 xgboost 和 lightgbm](https://www.cxybb.com/article/weixin_41411460/112358379)

[mac 安装 lightgbm 踩坑心得，亲测有效！](https://blog.csdn.net/weixin_32087115/article/details/81489627)

[MAC 上 使用 lightgbm 遇到 image not found 解决办法总结](https://blog.51cto.com/mapengfei/2476367)

[杂记-Macbook Pro M1 芯片能玩深度学习吗？](https://blog.csdn.net/jorg_zhao/article/details/109906053)