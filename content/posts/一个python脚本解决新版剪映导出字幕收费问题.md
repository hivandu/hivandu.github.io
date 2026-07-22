---
title: "一个python脚本解决新版剪映导出字幕收费问题"
date: 2024-07-24T07:03:00+08:00
draft: false
summary: "> 如果你是希望我能完全解决剪映收费问题，我无法帮你；"
---

>
> 两个文件，可生成不带时间线的纯文案，MD 格式，也可以生成带时间线的 SRT 文件。
>
> 因为剪映国内版对 JSON 文件进行了加密，所以请选择国际版 Cutcap，无法接受的请不要购买。



剪映在更新版本之后，原本的生成字幕功能变成 VIP 专享了，在最新的 6.1 版本中，甚至是 SVIP 专享。有点「鱼塘养肥了可以宰了」的感觉。说实在的，虽然剪映本身并没有声称过自己是免费软件，但是其背后的算法也是这么多用户为其贡献资源才慢慢完善的。

![image-20240723160632762](https://cdn.jsdelivr.net/gh/hivandu/notes/img/202407231611185.png)

好吧，我们要知道，基本百分之 90 以上的用户都在使用这个字幕功能为自己的视频添加字幕，否则多数口播博主的那个发音，估计三四遍都不知道你在讲什么。更何况很大一部分人生成字幕后都是要交给 AI 再读一次的。估计字节也是看准了这部分人不得不用，所以才敢这样。

既然无法导出带字幕的视频了，那我们换个方式吧，使用其他方式保存字幕之后，咱们再在 Final Cut 或者 Premiere 中去做，又或者，如果你是导入自己的字幕，剪映中依然是可以做的。那问题就好解决了，既然你剪映可以生成字幕，只是无法导出带字幕的视频或者导出字幕，那我在你生成之后获取源文件不就好了。

于是就有了我以下的自力更生的方法。

![image-20240723160342740](https://cdn.jsdelivr.net/gh/hivandu/notes/img/202407231611187.png)

![image-20240723160440991](https://cdn.jsdelivr.net/gh/hivandu/notes/img/202407231611188.png)

![image-20240723160517869](https://cdn.jsdelivr.net/gh/hivandu/notes/img/202407231611189.png)

![image-20240723160543909](https://cdn.jsdelivr.net/gh/hivandu/notes/img/202407231611190.png)

![image-20240724000355040](https://cdn.jsdelivr.net/gh/hivandu/notes/img/202407240004183.png)

![image-20240724000301004](https://cdn.jsdelivr.net/gh/hivandu/notes/img/202407240004184.png)

![image-20240724000315995](https://cdn.jsdelivr.net/gh/hivandu/notes/img/202407240004185.png)

用法也很简单，直接在终端执行: `python json2md.py`, 或者 `python json2srt.py`即可，这两个文件一个是转为不带时间线的 MD 格式文件，一个是转为带时间线的 SRT 文件。

因为不太清楚剪映中对于时间的计算规则，所以规则是摸索的，最后造成的结果就是大概每一块字幕和原字幕大概误差在 1~2 帧左右，实际使用并无影响。

目前本程序只是选择读取文件的目录和保存文件的目录，即便你的剪映修改了草稿目录照样可以执行。默认为剪映默认文件目录。

有需求的朋友，[可自行到此处下载](https://mp.weixin.qq.com/s/8hD0tQ7rwfHPGUX-DxE8ww)。