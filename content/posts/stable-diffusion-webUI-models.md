---
title: "玩转 Stable Diffusion WebUI 各类模型"
date: 2023-03-09T16:17:18+08:00
draft: false
tags: []
summary: "Stable Diffusion WebUI 最有意思的地方不是在安装好之后生成图像，而是各种各样的模型。

提前警告：如果你的硬盘空间不够大的话，还是不要随便玩模型了，随随便便就是好几 G，又得甚至于 10 多个 G。"
---

目前我仅留了最常用的 SD V1.5 和 SD V2.1 两个模型，大小为 13G。

![notion image](https://qiniu.hivan.me/picGo/20230601161901.png?imgNote)

另外还需要说明一点，就是我曾经测试过用 NAS 来存储模型使用，完全不能用，暂时没有时间具体去研究到底什么原因。只有老老实实的继续在本地硬盘上跑。所以 NAS 上存了大量模型，真需要用到的时候再复制过来。

写这篇文章也是因为近期玩模型过程中打算整理一波，一是方便自己，二么也算是对其他小伙伴做些贡献。

 

Stable Diffusion 各种模型层出不穷，要说完估计需要费一番功夫，所以我摒弃其他小模型，只整理收集大模型，就是 ckpt 和 safetensors。如果你也打算跟着我一起玩模型但是还未安装，可以先参看我之前的文章：

在 Apple Silicon M1/M2 Mac 上安装和运行 Stable Diffusion

说实话，我找了好多关于如何在 M1/M2 上安装和运行 Stable Diffusion 的教程和帖子，发现相互之间借鉴的不少，但是能用的确实没几个。 寻找一番后，发现其实没那么复杂。也不知道为什么网上的那么多教程搞得那么复杂，又是这个又是那个的一大堆，简单实现的方式有好几种：

https://www.hivan.me/How%20to%20install%20and%20run%20Stable%20Diffusion%20on%20Apple%20Silicon

![在 Apple Silicon M1/M2 Mac 上安装和运行 Stable Diffusion](https://qiniu.hivan.me/picGo/20230601161911.png?imgNote)

 

还是先从最基础的模型开始：

## Stable Diffusion

其他多数模型基本上都是从这个基础模型上再次训练得到的。

### Stable Diffusion v2.1

1. SDv2.1 提升了人物生成能力，因为 SDv2.0 大量增加了风景、建筑物和动物的数据集，减少了人物的学习量。

1. SDv2.1 提高了 NSFW 过滤器准确度，因为 SDv2.0 的成人过滤器过滤的太狠，错误判定很多

1. 即使是极端长宽比的图像也能顺利生成。

1. 解剖学的身体和手（特别是手掌）的描写精度提高。

#### 512 X 512 model :

stabilityai/stable-diffusion-2-1-base · Hugging Face

We’re on a journey to advance and democratize artificial intelligence through open source and open science.

https://huggingface.co/stabilityai/stable-diffusion-2-1-base

![stabilityai/stable-diffusion-2-1-base · Hugging Face](https://thumbnails.huggingface.co/social-thumbnails/models/stabilityai/stable-diffusion-2-1-base.png)

#### 768 X 768 model:

stabilityai/stable-diffusion-2-1 · Hugging Face

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/stabilityai/stable-diffusion-2-1

![stabilityai/stable-diffusion-2-1 · Hugging Face](https://thumbnails.huggingface.co/social-thumbnails/models/stabilityai/stable-diffusion-2-1.png)

#### img2img model

stabilityai/stable-diffusion-2-depth · Hugging Face

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/stabilityai/stable-diffusion-2-depth

![stabilityai/stable-diffusion-2-depth · Hugging Face](https://thumbnails.huggingface.co/social-thumbnails/models/stabilityai/stable-diffusion-2-depth.png)

#### 重绘 model

stabilityai/stable-diffusion-2-inpainting · Hugging Face

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/stabilityai/stable-diffusion-2-inpainting

![stabilityai/stable-diffusion-2-inpainting · Hugging Face](https://thumbnails.huggingface.co/social-thumbnails/models/stabilityai/stable-diffusion-2-inpainting.png)

#### 超分 model

stabilityai/stable-diffusion-x4-upscaler · Hugging Face

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler

![stabilityai/stable-diffusion-x4-upscaler · Hugging Face](https://thumbnails.huggingface.co/social-thumbnails/models/stabilityai/stable-diffusion-x4-upscaler.png)

 

### Stable Diffusion V 1.5

runwayml/stable-diffusion-v1-5 at main

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main

![runwayml/stable-diffusion-v1-5 at main](https://thumbnails.huggingface.co/social-thumbnails/models/runwayml/stable-diffusion-v1-5.png)

 

### Stable Diffusion V 1.4

CompVis/stable-diffusion-v-1-4-original · Hugging Face

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/CompVis/stable-diffusion-v-1-4-original

![CompVis/stable-diffusion-v-1-4-original · Hugging Face](https://thumbnails.huggingface.co/social-thumbnails/models/CompVis/stable-diffusion-v-1-4-original.png)

 

## NovelAI

大名鼎鼎的 NovelAI，属于商业泄露模型。经过人在回路精细微调，可以生成高质量的二次元图像。但是千万时刻记得这个可是商用泄露模型，要注意避免法律风险：

pub-2fdef7a2969f43289c42ac5ae3412fd4.r2.dev

https://pub-2fdef7a2969f43289c42ac5ae3412fd4.r2.dev/animefull-latest.tar

## Waifu Diffusion

基于 Stable Diffusion 模型训练得到，增加了动漫及人物训练得到的模型，基本平时各种公开场合看到 WD 就是他。

WD 和 NovelAI 模型有些同质化，但是 NovelAI 实际是商用模型泄露，在某些使用情况下是有风险的。而 WD 不是，不过也不是说他绝对安全，毕竟 WD 也使用 Danbooru 进行学习，所以如果你关心这个需要注意一点。

### Waifu Diffusion V1.5

这个模型使用是需要一个 yaml 文件的，究其原因是这个模型是基于 SD V2 得出的，需要把和 Model 同名的 yaml 文件放在模型所在的文件夹下，目前 1.5 模型是 beta2 版本，持续迭代 ing…

#### Waifu Diffusion v1.5 beta

waifu-diffusion/wd-1-5-beta · Hugging Face

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/waifu-diffusion/wd-1-5-beta

![waifu-diffusion/wd-1-5-beta · Hugging Face](https://thumbnails.huggingface.co/social-thumbnails/models/waifu-diffusion/wd-1-5-beta.png)

#### VAE(1.4 VAE 通用)

vae/kl-f8-anime2.ckpt · hakurei/waifu-diffusion-v1-4 at main

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/hakurei/waifu-diffusion-v1-4/blob/main/vae/kl-f8-anime2.ckpt

![vae/kl-f8-anime2.ckpt · hakurei/waifu-diffusion-v1-4 at main](https://thumbnails.huggingface.co/social-thumbnails/models/hakurei/waifu-diffusion-v1-4.png)

#### YAML

waifu-diffusion/wd-1-5-beta2 at main

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/waifu-diffusion/wd-1-5-beta2/tree/main/checkpoints

![waifu-diffusion/wd-1-5-beta2 at main](https://thumbnails.huggingface.co/social-thumbnails/models/waifu-diffusion/wd-1-5-beta2.png)

 

### Waifu Diffusion V1.4

和 1.5 版本一样，基于 SD V2 得到的，依然需要下载 yaml 文件放在 model 同文件夹下。

#### Waifu Diffusion V 1.4

wd-1-4-anime_e1.ckpt · hakurei/waifu-diffusion-v1-4 at main

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/hakurei/waifu-diffusion-v1-4/blob/main/wd-1-4-anime_e1.ckpt

![wd-1-4-anime_e1.ckpt · hakurei/waifu-diffusion-v1-4 at main](https://thumbnails.huggingface.co/social-thumbnails/models/hakurei/waifu-diffusion-v1-4.png)

wd-1-4-anime_e2.ckpt · hakurei/waifu-diffusion-v1-4 at main

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/hakurei/waifu-diffusion-v1-4/blob/main/wd-1-4-anime_e2.ckpt

![wd-1-4-anime_e2.ckpt · hakurei/waifu-diffusion-v1-4 at main](https://thumbnails.huggingface.co/social-thumbnails/models/hakurei/waifu-diffusion-v1-4.png)

#### VAE(1.5 通用）

vae/kl-f8-anime2.ckpt · hakurei/waifu-diffusion-v1-4 at main

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/hakurei/waifu-diffusion-v1-4/blob/main/vae/kl-f8-anime2.ckpt

![vae/kl-f8-anime2.ckpt · hakurei/waifu-diffusion-v1-4 at main](https://thumbnails.huggingface.co/social-thumbnails/models/hakurei/waifu-diffusion-v1-4.png)

#### YAML

e2 和 e1 是通用的，但是需要改名

hakurei/waifu-diffusion-v1-4 at main

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/hakurei/waifu-diffusion-v1-4/tree/main

![hakurei/waifu-diffusion-v1-4 at main](https://thumbnails.huggingface.co/social-thumbnails/models/hakurei/waifu-diffusion-v1-4.png)

## ***\*Elysium Anime\****

生成偏真实风格的动漫图片，风格比较偏向西式，光影还不错。

模型推荐写下面这些负面提示，可有效提升质量。

```bash
lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry
```

 

### Elysium_V1

偏真实风的模型，手画的还不错，模型底稿基本是以西方人为主，所以生成的脸也偏西方人。

Elysium_V1.ckpt · hesw23168/SD-Elysium-Model at main

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/hesw23168/SD-Elysium-Model/blob/main/Elysium_V1.ckpt

![Elysium_V1.ckpt · hesw23168/SD-Elysium-Model at main](https://thumbnails.huggingface.co/social-thumbnails/models/hesw23168/SD-Elysium-Model.png)

### ***\*SD_Elysium_Kuro_Model\****

与 Anything 4.0、WD 1.4 等合并后经过微调的二次元用模型。已经包含 WD 的“kl-f8-anime2”VAE 文件，因此**无需使用额外的 VAE 文件**

Elysium_Kuro_Anime_V1.safetensors · hesw23168/SD_Elysium_Kuro_Model at main

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/hesw23168/SD_Elysium_Kuro_Model/blob/main/Elysium_Kuro_Anime_V1.safetensors

![Elysium_Kuro_Anime_V1.safetensors · hesw23168/SD_Elysium_Kuro_Model at main](https://thumbnails.huggingface.co/social-thumbnails/models/hesw23168/SD_Elysium_Kuro_Model.png)

### ***\*Elysium_Anime_V3\****

动漫的附加学习模型，NSFW 化相当严重，有更清晰的轮廓和轻微的三维效果。基于 Elysium_V1

Elysium_Anime_V3.safetensors · hesw23168/SD-Elysium-Model at main

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/hesw23168/SD-Elysium-Model/blob/main/Elysium_Anime_V3.safetensors

![Elysium_Anime_V3.safetensors · hesw23168/SD-Elysium-Model at main](https://thumbnails.huggingface.co/social-thumbnails/models/hesw23168/SD-Elysium-Model.png)

## ***\*Anything 系列\****

Anything 是个神奇的二次元模型，据说是基于几十种模型融合+未知图片训练而来，随便写几个提示，就能到的不错的结果。不过这个模型整个就是一团混沌，实际训练模型，过程，方法，作者全部都是未知的。模型容易过拟合，非专业人士，请不要在此基础上训练模型。

### **Anything v3.0**

“应该”是基于 NAI 模型+WD+SD 等几十种模型+手部图片强化训练得出的。实际训练模型，过程，方法，作者全部都是未知的。如果没有.vae.pt，图片整体颜色浓度（饱和度）会更很浅。PS：Anything v3.0 的 .vae.pt 文件可以用于 NAI。

- Anything V3.0 fp16: magnet:?xt=urn:btih/:45cd353ac4fa87098db5e3a6a349539710a3a1f5&dn=Anything-V3.0-fp16.zip

- Anything v3.0 fp32: magnet:?xt=urn:btih/:d9db662ab5ace77004b3348c23c9381380c27156&dn=Anything-V3.0-fp32.zip

- Anything v3.0 full-ema: magnet:?xt=urn:btih/:80460036625fb61dce4bc6e7dab744744309a2a0&dn=Anything-V3.0-fullema.zip

huggingface.co

https://huggingface.co/Linaqruf/anything-v3-better-vae/tree/main

### **Anything v4**

自称是 Anything 最新版本的模型，实际一切都是未知的。仅需几个提示即可生成详细的 2D 插图的能力以及使用 danbooru 标签的能力。整体比过拟合的 v3 更自然，人物姿势等更容易操作。

anything-v4.0-pruned.safetensors · andite/anything-v4.0 at main

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/andite/anything-v4.0/blob/main/anything-v4.0-pruned.safetensors

![anything-v4.0-pruned.safetensors · andite/anything-v4.0 at main](https://thumbnails.huggingface.co/social-thumbnails/models/andite/anything-v4.0.png)

### **Anything v4.5**

貌似是 Anything v4 的进化，但实际一切都是未知的。比 v4 画风更柔和一点。

anything-v4.5-pruned.safetensors · andite/anything-v4.0 at main

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/andite/anything-v4.0/blob/main/anything-v4.5-pruned.safetensors

![anything-v4.5-pruned.safetensors · andite/anything-v4.0 at main](https://thumbnails.huggingface.co/social-thumbnails/models/andite/anything-v4.0.png)

## **Zeipher**

生成更符合真人解剖结构的真人模型，训练集以女性图像为主官方网站是 [https://ai.zeipher.com](https://ai.zeipher.com/)，已经关闭。**请不要用真人模型画明星和未成年的 NSFW 内容，不然你可能会遇到很麻烦的法律问题**

### **F222**

f222.safetensors · acheong08/f222 at main

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/acheong08/f222/blob/main/f222.safetensors

![f222.safetensors · acheong08/f222 at main](https://thumbnails.huggingface.co/social-thumbnails/models/acheong08/f222.png)

### **F111**

f111.ckpt · Reachout/F111 at main

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/Reachout/F111/blob/main/f111.ckpt

![f111.ckpt · Reachout/F111 at main](https://thumbnails.huggingface.co/social-thumbnails/models/Reachout/F111.png)

## **3DKX**

因为 Zeipher 官方已经 GG，这是热心网友创建的衍生 3DKX 模型如果你想让你的 3D 角色有一张更“二次元”的脸，提示词最开始写 “3d cartoon of”，或者如果你想要经典的 3D 渲染外观，写“a 3d render of”高分辨率模型，推荐分辨率为 1152 x 768 或更高

### **3DKX_1.0b**

f111.ckpt · Reachout/F111 at main

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/Reachout/F111/blob/main/f111.ckpt

![f111.ckpt · Reachout/F111 at main](https://thumbnails.huggingface.co/social-thumbnails/models/Reachout/F111.png)

## **R34**

从网站“rule34.xxx”的 150,000 张图像中进行训练。rule34.xxx 几乎全是 NSFW 图片，所以你懂的

### **r34_e4**

1.99 GB file on MEGA

![1.99 GB file on MEGA](https://mega.nz/favicon.ico?v=3)

https://mega.nz/file/yJgDUCzA#zOD2yeE6QLBqPEjEpIi2b4FWOlb64yVUveOd_eW6teI

![1.99 GB file on MEGA](https://mega.nz/rich-file.png)

磁力链接：magnet:?xt=urn:btih/:ed9f0e3f849d7119107ef4e072c6abeb129e1a51&dn=r34_e4.ckpt

## **EVT pixiv 排行榜模型**

基于 pixiv 排行图片训练，夹杂有部分 R18 排行图片

### **Evt_V4_e10_ema**

Evt_V4_e10_ema.safetensors · haor/Evt_V4-preview at main

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/haor/Evt_V4-preview/blob/main/Evt_V4_e10_ema.safetensors

![Evt_V4_e10_ema.safetensors · haor/Evt_V4-preview at main](https://thumbnails.huggingface.co/social-thumbnails/models/haor/Evt_V4-preview.png)

### **EVT_V3**

haor/Evt_V3 · Hugging Face

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/haor/Evt_V3

![haor/Evt_V3 · Hugging Face](https://thumbnails.huggingface.co/social-thumbnails/models/haor/Evt_V3.png)

### **EVT_V2**

haor/Evt_V2 · Hugging Face

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/haor/Evt_V2

![haor/Evt_V2 · Hugging Face](https://thumbnails.huggingface.co/social-thumbnails/models/haor/Evt_V2.png)

 

## **Basil_mix**

逼真的真人模型，基于亚洲风格训练，支持 Danbur 标签提示词需要加载 VAE，不然画面色彩浓度和边缘会很淡提示词应尽可能简单不要堆砌大量质量标签和负面提示，不然会适得其反。**请不要用真人模型画明星和未成年的 NSFW 内容，不然你可能会遇到很麻烦的法律问题**

#### **basil_mix**

basil mix.ckpt · nuigurumi/basil_mix at main

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/nuigurumi/basil_mix/blob/main/basil%20mix.ckpt

![basil mix.ckpt · nuigurumi/basil_mix at main](https://thumbnails.huggingface.co/social-thumbnails/models/nuigurumi/basil_mix.png)

#### VAE

vae-ft-mse-840000-ema-pruned.ckpt · stabilityai/sd-vae-ft-mse-original at main

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.ckpt

![vae-ft-mse-840000-ema-pruned.ckpt · stabilityai/sd-vae-ft-mse-original at main](https://thumbnails.huggingface.co/social-thumbnails/models/stabilityai/sd-vae-ft-mse-original.png)

## **Chillout Mix**

逼真的真人模型，基于亚洲风格训练，支持 Danbur 标签提示词**请不要用真人模型画明星和未成年的 NSFW 内容，不然你可能会遇到很麻烦的法律问题**

#### **chillout mix _ NiPruned Fp32 Fix**

chilloutmix_NiPrunedFp32Fix.safetensors · Inzamam567/useless_Chillout_mix at main

We’re on a journey to advance and democratize artificial intelligence through open source and open science.


https://huggingface.co/Inzamam567/useless_Chillout_mix/blob/main/chilloutmix_NiPrunedFp32Fix.safetensors

## **Uber Realistic Porn Merge**

如名字所说，逼真的真人 Porn 模型，简称 URPM 模型**请不要用真人模型画明星和未成年的 NSFW 内容，不然你可能会遇到很麻烦的法律问题**

#### **Uber Realistic Porn Merge**

Uber Realistic Porn Merge (URPM) | Stable Diffusion Checkpoint | Civitai

For early access builds and to support daily work on URPM, please check out my patreon! https://www.patreon.com/uber_realistic_porn_merge , or disc...

![Uber Realistic Porn Merge (URPM) | Stable Diffusion Checkpoint | Civitai](https://civitai.com/favicon.ico)

https://civitai.com/models/2661/uber-realistic-porn-merge-urpm