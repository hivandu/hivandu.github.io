---
title: "关于设备转向后的自适应"
date: 2016-07-27T12:50:44+08:00
draft: false
tags: []
---

关于移动端的适配，都知道其实 rem 是比较好的一个适配方案，但是 rem 是根据根目录的字体大小来调解的，那么，我们在做网页的时候，屏幕旋转后，能否让根目录的字体跟着变化呢？

先上代码：

```javascript
$(function(){
  var size = $(window).width() / 25;
  $('html').css('font-size': size);
});
```

这样在 css 中用 rem 单位是没什么问题，但是如果屏幕旋转之后，你就会发现，真的不能看了就。原因就是屏幕旋转以后，根上的字体并没有随之变化。

所以我们来加上

```javascript
// 监视设备方向
window.addEventListener("orientationchange", function() {
  media();
}, false);

function media(argument) {
  // 因为获取尺寸出错，需要延迟获取
  setTimeout(function(){
    var size = $(window).width() / 25;
    console.log('the device size: '+size);
    $('html').css('font-size', size);
  }, 200);  
}
```
