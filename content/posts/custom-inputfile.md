---
title: "自定义文件上传框"
date: 2015-07-27T12:50:44+08:00
draft: false
---

其实这根本就不值得写出来，只是可能前几步大家都做了，只是最后一步就忽略了。
![][image-1]
我们在自定义`input:file`的时候，一般来说都是外边包一层，里边在写一个`<input type="file">`, 然后将其透明值设置成`0`,然后再定义外层的样式来达到自定义的目的。

<!--more-->

HTML：

	<div class="upfileOutWrap">
	  <div class="upfileWrap"><input type="file"></div>
	  <div class="upfileBG">upload image</div>
	</div>

**CSS：**

	.upfileOutWrap { 
	  cursor: pointer; 
	  width: 199px;  height: 42px; 
	  line-height: 42px; 
	  position: relative;
	}
	.upfileWrap{
	  width: 100%;  height: 100%; 
	  position: absolute; 
	  top:-1; left: -1; 
	  z-index:2;
	}
	.upfileWrap input{
	  opacity: 0; filter: alpha(opacity=0); 
	  cursor: pointer; 
	  width: 100%; height: 100%;
	  font-size: 32px;
	}
	.upfileBG{
	  width:100%; height:100%; 
	  background: url(./images/upload.png) no-repeat; 
	  font-size: 14px; 
	  color: white; 
	  position: absolute; 
	  top:-1; left: -1; 
	  padding-left:10px; 
	  z-index:1;
	}

![][image-2]

可是这个时候还是有点问题，就是万恶的 IE 下边。

IE 下边的`input`标签默认都是有光标的，`:file`也不例外，而且 IE 下边必须要点击”Browse”或者双击`input`输入框才会有效果。那么这个时候在 IE 下就会出现如图的莫名其妙的问题，注意左边的光标，并且还需要双击才会弹出文件选择窗口。
![][image-3]

这个时候如果你把 input 透明度设置成 100 显示出来，就会发现原来是这样的。

![][image-4]

所以这个时候，如果是其他标准浏览器，那么设置好 input 的高宽就搞定了，而 IE 下边，还必须考虑如何让”Browse”按钮能铺满我们所自定的 div 样式。这样我们才能实现 IE 下不出现光标，而且单击弹出文件选择窗口。

这个时候，看似毫无办法，其实我们可以选择增加字体的大小。当字体变成`32px`的时候，就是这个样子的。
![][image-5]

好了，这样我们就搞定了，将`input:file` 继续设置为完全透明。那个可恶的光标不见了，我们也可以实现 IE 下单击。当然，字体到底用多大的，要视你自己定义的视觉效果来看，自己调试吧。

** Final CSS: **

	.upfileOutWrap { 
	  cursor: pointer; 
	  width: 199px;  height: 42px; 
	  line-height: 42px; 
	  position: relative;
	}
	.upfileWrap{
	  width: 100%;  height: 100%; 
	  position: absolute; 
	  top:-1; left: -1; 
	  z-index:2;
	}
	.upfileWrap input{
	  opacity: 0; filter: alpha(opacity=0); 
	  cursor: pointer; 
	  width: 100%; height: 100%;
	}
	.upfileBG{
	  width:100%; height:100%; 
	  background: url(./images/upload.png) no-repeat; 
	  font-size: 14px; 
	  color: white; 
	  position: absolute; 
	  top:-1; left: -1; 
	  padding-left:10px; 
	  z-index:1;
	}

[image-1]:	https://farm6.staticflickr.com/5196/14320298586_2c05c821ac_o_d.png

[image-2]:	https://farm4.staticflickr.com/3874/14156755400_e1e91a66b6_o_d.png
[image-3]:	https://farm4.staticflickr.com/3903/14343379175_89590a504f_o_d.png
[image-4]:	https://farm4.staticflickr.com/3920/14342594154_1cb4af92b4_o_d.png
[image-5]:	https://farm4.staticflickr.com/3926/14156914197_ccb8fd7e43_o_d.png