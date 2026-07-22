---
title: "解决 Mac M1 原生 Photoshop 找不到 CEP 扩展面板"
date: 2022-03-19T16:04:32+08:00
draft: false
tags: []
summary: "研究这个问题, 也属于是撞上了!"
---

我现在使用的 PS 版本为:

![notion image](https://qiniu.hivan.me/picGo/20230601160612.png?imgNote)

在 Photoshop 内, 我画画时一直使用的是第三方的色轮插件 Coolorus, 长这样:

![notion image](https://qiniu.hivan.me/picGo/20230601160617.png?imgNote)

好用与否, 可以说, 谁用谁知道.

前些日子发现 Photoshop2022 有 M1 原生版本, 不用再使用转译版本, 我心想, 应该速度上会快很多吧!

兴奋的更新完后才发现, Coolorus 面板无论如何找不到了, 原本应该在窗口下的“扩展(旧版)”菜单也找不到.

![notion image](https://qiniu.hivan.me/picGo/20230601160623.png?imgNote)

无论怎么折腾都不行, 而且设置面板里的增效工具也是灰色无法设置:

![notion image](https://qiniu.hivan.me/picGo/20230601160627.jpg?imgNote)

正当我心灰意冷准备返回 2021 时, 忽然想到, Adobe 早就在 PS 中启用 UXP 插件了, 而 CEP 插件因为历史原因一直也无法完全取消, 那么既然是早就做的事情, 为什么 2022 版本里全给抹杀了呢, 重点是, 面板里设置项虽然不可点击, 但是还在? 问题应该不是出在版本上, 而是出在 M1 原生的问题上, 我试着去设置了转译, 再次重新打开 PS, 果然不出所料, 扩展(旧版)菜单又回来了.

![notion image](https://qiniu.hivan.me/picGo/20230601160632.png?imgNote)

![notion image](https://qiniu.hivan.me/picGo/20230601160637.jpg?imgNote)

好吧, 这回知道是怎么回事了, 也就好解决了.

方法一: 返回 PS 22.21 版本, 简单粗暴

方法二: 讲 PS2022 设置为 Rosetta 转译打开方式, 同样简单粗暴!

回答一下可能大家问到的问题:

1. 速度上, 感觉不出有什么太大的变化

1. 是的, Coolous 照样无法使用, 我快崩溃了, 正在纠结到底是放弃 Coolous 使用 Photoshop 原始色轮, 还是回到 2021

反正问题是这么个问题, 解决方案也有了!大家自行抉择吧!