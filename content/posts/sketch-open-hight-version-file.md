---
title: "sketch 中打开高版本文件"
date: 2024-01-01T00:00:00+08:00
draft: false
---

sketch 也不知道什么时候开始年费化了，也不能打开高版本文件了。（妈蛋）

据说是为了促进销量和保护版本。

打开包文件，然后打开包内的`meta.json`

替换头部:

```json
{"commit":"335a30073fcb2dc64a0abd6148ae147d694c887d","appVersion":"43.1","build":39012
```

替换尾部

```json
"commit":"335a30073fcb2dc64a0abd6148ae147d694c887d","build":39012,"appVersion":"43.1","variant":"NONAPPSTORE","version":88},"version":88,"saveHistory":["NONAPPSTORE.39012"],"autosaved":0,"variant":"NONAPPSTORE"}
```

这里实际有几个 key:`commit`, `appVersion`, `build`, `version`,`NONAPPSTORE`

value 替换成相应的值就 OK 了。



