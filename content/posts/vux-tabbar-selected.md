---
title: "vux 更改 Tabbar 选中状态"
date: 2017-09-03T00:00:00+08:00
draft: false
---

在 vux 的文档和示例中，都没有明确的说明 tabbar 上 v-model 的使用

文档中将`v-model`说明放在了 TabbarItem 示例下，但是其实这个应该是放在`Tabbar`上

```javascript
<template>
    <router-view class="view" v-on:changeTab="changeTab"></router-view>
    <tabbar v-model="index">
        <tabbar-item></tabbar-item>
        ...
        <tabbar-item></tabbar-item>
    </tabbar>
</template>
<script>
data(){
    return{
        index:0,
        ...
    }
}
methods:{
    changeTab(num){
        ...
        this.index = num;
        ...
    }
}
</scirpt>
```

然后子组件中调用

```javascript
mounted(){
    this.$emit('changeTab', 2)
}
```

这样就便于在不同的组件内都可以更改 Tabbar 选中状态

