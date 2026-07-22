---
title: "vue 2.0 自定义 filter 并挂载到全局使用"
date: 2017-10-21T00:00:00+08:00
draft: false
summary: "vue 2.0 开始，取消默认 filter, 需要自定义。"
---

而自定义之后每次在需要使用的组件内引用也确实蛮麻烦的。

所以我们就来将定义的 filter 挂载到全局使用。

[vue2.0 filter 相关文档][1]

- 定义
- 引用
- 挂载
- 使用

**/src/filters/**
- format.js

```javascript
export default function(val){
    ...
}
```


- index.js

```javascript
import format from "./format";

export default{
  format: format,
}
```

**/src/**
- main.js

``` javascript
...

import commonFiltes from './filters/index'

Object.keys(commonFiltes).forEach(function (key, index, arr) {
  Vue.filter(key, commonFiltes[key]);
})

...
```

**/src/components/**
- xxx.vue

```html
<template>
...
<div>{{ data | format }}</div>
</template>
<script>
...
</script>
```



[1]:https://cn.vuejs.org/v2/guide/syntax.html#Filters