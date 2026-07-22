---
title: "使用 Python 构建颜色选择器"
date: 2024-10-18T07:03:00+08:00
draft: false
summary: "> 创建一个从图像像素中选择 RGB 通道的工具"
---

![不同种类的彩色鸟类](https://cdn.jsdelivr.net/gh/hivandu/notes/img/20241011193148.png)

我知道，市面上有很多种颜色选择工具。但我认为，你会发现使用一种直接在笔记本中工作的工具更有优势。此工具也是完全可定制的。

我们将构建两个颜色选择器：

- 简单选择器——从一张图片中选择一种颜色
- 复杂选择器——从多幅图像中选择颜色列表并显示颜色

最后，我们讨论一下数据科学中的一些应用。这些是我使用颜色选择器的方法。

## 导入

让我们深入研究代码。你还可以在 GitHub[^1] 上找到完整的项目。

首先，我们有一些导入。我们有标准包（第 2-3 行）。`mpimg` 用于加载图像，`pyperclip` 用于将字符串保存到剪贴板，`glob` 用于处理文件路径。确保已安装所有这些。

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pyperclip
import random
import glob
```

我们将从不同的鸟类图像中挑选颜色。你可以在文章的封面图片中看到一些示例。我们在给定的目录（第 2 行）中加载所有图像路径（第 3 行）。

```python
# 数据集
data = dp + 'birds/'
img_path = glob.glob(data + '*.jpg')
```

## 简单的颜色选择器

在图 1 中，你可以看到我们的第一个选择器正在运行。每次我们点击图像中的某个位置时，该像素的 RGB 通道都会保存到剪贴板中。然后我们可以将该值直接粘贴到笔记本中。

![图 1：使用 Python 创建的简单颜色选择器](https://cdn.jsdelivr.net/gh/hivandu/notes/img/Colour_1.gif)

要创建此选择器，我们从 `onclick` 函数开始。每次单击图像时都会运行此函数。我们获取单击的 x 和 y 坐标（第 5-6 行）。然后，我们获取该坐标处像素的 RGB 通道（第 9 行）。最后，我们将这些通道作为字符串保存到剪贴板（第 12 行）。

```python
def onclick(event):
    global img

    # get x, y of click
    x = round(event.xdata)
    y = round(event.ydata)

    # Get RGB values
    rgb = img[y][x]

    # save to clip board
    pyperclip.copy(str(rgb))
```

要使用此函数，我们首先使用 matplotlib 创建一个图形（第 3 行）。然后，我们使用 `mpl_connect` 函数向该图形添加交互功能（第 10 行）。你可以看到，我们已将 `onclick` 函数作为参数传入。我们还加载了其中一张鸟类图像（第 6-7 行）并显示它（第 11 行）。

```python
global img

fig = plt.figure(figsize=(5,5))

# Load image and add count
path = img_path[-1]
img = mpimg.imread(path)

# Add an interactive widget to figure
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.imshow(img)
plt.show()
```

另一件需要注意的事情是使用全局变量（第 1 行）。这允许在 `onclick` 函数中更新这些变量。这是将图像作为参数传递的替代方法。我们还可以添加一行 `%matplotlib notebook`。这会将图形保留在笔记本中。

## 复杂的颜色选择器

现在让我们来调整一下。在图 2 中，我们现在有一个图像框（左）和颜色框（右）。我们现在可以看到我们点击的像素的颜色，并遍历多个图像。另外，请注意颜色框左上角的红色数字。我们能够将颜色保存到列表中，并且每次这样做时，这个数字都会更新。

![图 2：使用 Python 创建的复杂颜色选择器](https://cdn.jsdelivr.net/gh/hivandu/notes/img/Colour_3.gif)

再次，我们从 `onclick` 函数开始。这类似于之前的颜色选择器。主要区别在于我们现在运行函数 `change_choice`，而不是保存 RGB 通道。我们还更新了一个全局 rgb 变量。这样它就可以被下面的其他函数访问。

```python
def onclick(event):
    global img
    global rgb
    
    # get x,y of click
    x = round(event.xdata)
    y = round(event.ydata)
    
    # get RGB values
    rgb = img[y][x]
    
    #Update second plot with colour
    change_choice()
```

我们有一个函数 `onpress`，它会在按下键盘时运行。我们首先获取键（第 6 行）。接下来发生的事情取决于按下了什么键：

- n（下一步）：我们运行 `change_image` 函数
- c（复制）：我们将 RGB 通道保存到剪贴板（第 13 行）和颜色列表（第 16 行）。我们还运行 `change_choice` 函数。

请记住，全局 rgb 变量在 `onclick` 函数运行时更新。这意味着当我们按下 “`c`” 时，最近一次点击的 RGB 通道将被保存。

```python
def onpress(event):
    global rgb
    global colours
    
    #Get key 
    key = event.key

    if key == 'n':
        change_image()
        
    elif key == 'c':
         # save to clip board
        pyperclip.copy(str(rgb))
        
        # add to list of colours
        colours.append(rgb)
        
        change_choice()
```

`change_choice` 用于更新颜色框。要创建此框，我们使用与图像框相同的尺寸（第 13 行）。颜色框中的每个像素都将具有当前全局 rgb 值的 RGB 通道（第14 行）。我们还删除当前计数（第 9-10 行），然后更新它（第 18 行）。为此，我们使用已保存的颜色列表的长度。

```python
def change_choice():
    global img
    global ax
    global colours
    global rgb
    
    # remove previous count
    for txt in ax[1].texts:
        txt.set_visible(False)
    
    # create array of colour choice
    dims = np.shape(img)
    col = np.array([[rgb]*dims[0]]*dims[1])
    ax[1].imshow(col)
    
    # update colour count
    ax[1].text(0, 15, len(colours),color='r',size=20)
    
    plt.show()
```

`change_choice` 函数运行时有两个动作：

- 当我们点击图像时，通过 `onclick` 函数。在这种情况下，全局 rgb 会更新，并且框中的颜色会发生变化。
- 当我们按下 “`c`” 时，通过 `onpress` 函数。这里颜色列表的长度增加了，红色的数字也会改变。

最后，我们有 `change_image` 函数。每当我们按 “`n`” 时，它都会用来更新图像框。我们首先关闭所有现有图（第 8 行）。然后创建一个新图（第 10 行），并为其添加单击（第 13 行）和按下（第 14 行）功能。我们加载并显示一个随机的鸟类图像（第 17-20 行）。然后我们更新颜色框（第 24 行）。通过首先将全局 rgb 变量设置为 `[255,255,255]`，我们将框颜色设置为白色。

```python
def change_image():
    global img_path
    global img
    global ax
    global rgb
    
    # close all open plots
    plt.close('all')
    
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    
    # add an interactive widget to figure 
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    cid2 = fig.canvas.mpl_connect('key_press_event', onpress)

    # load random image
    path = random.choice(img_path)
    img = mpimg.imread(path)
    
    ax[0].imshow(img)
    
    # reset the colour window
    rgb = [255,255,255]
    change_choice()
```

我们可以通过运行 `change_image` 函数（第 12 行）来启动颜色选择器。请注意，我们现在在第 1 行有 `%matplotlib tk`。这将在笔记本外的窗口中打开颜色选择器。如果你尝试直接在笔记本中运行它，它实际上将不起作用。如果有人能解决这个问题，请在评论中告诉我 :)。

```python
%matplotlib tk
global img_path
global colours
colours = []

# load image paths
img_path = glob.glob(data + "*.jpg")

# start widget
change_image()
```

当你遍历图像并保存颜色时，颜色列表将会更新。图 3给出了此类列表的一个示例。这来自我们在图 2中看到的鸟类图像。

![图 3：Python RGB 颜色通道列表](https://cdn.jsdelivr.net/gh/hivandu/notes/img/20241011231228.png)

## 数据科学应用

我想与你分享这段代码，因为我发现它在我的数据科学之旅中很有用。在本文的其余部分，我们将讨论一些应用程序。

### 合并图表颜色

你可以说我是个完美主义者，但在展示工作时，我喜欢所有图表都采用相同的配色方案。问题是我倾向于使用多个 Python 包。在图 4中，你可以看到 matplotlib（左）和[SHAP](https://mp.weixin.qq.com/s/qdzJN-A23OdP4DHnW4tJrw) （右）使用的默认颜色之间的差异。

![图 4：使用 Python 创建的 Matplotlib 和 SHAP 图](https://miro.medium.com/v2/resize:fit:700/1*UhBx2FH9CWs8tAiMxaWzNQ.png)

使用第一个颜色选择器，我能够解决这个问题。我可以用 Python 保存这些图表、加载它们并选择它们的颜色。更新 matplotlib 图表很简单。或者，我们未来可以写一篇《[[如何自定义 SHAP 图]]》

> [Leonie Monigatti](https://medium.com/u/3a38da70d8dc?source=post_page-----55e8357539e7--------------------------------)有一个[关于的自定义 SHAP 图很好的教程](https://medium.com/towards-data-science/how-to-easily-customize-shap-plots-in-python-fdff9c0483f2)。

你可能还会发现下面的代码很有用。它将 RGB 通道转换为十六进制字符串。我发现有些包只接受它作为颜色参数。

```python
# 将 RGB 转换为十六进制
from colormap import rgb2hex
rgb2hex(134,94,58)
```

### 利用图像数据进行特征工程

第二个应用是我决定构建更复杂的颜色选择器的原因。我用它来创建机器学习的功能。在图 5 中，你可以看到橙色轨道如何与图像的其余部分隔离。轨道像素的颜色是使用颜色选择器获得的。

![图 5：使用图像数据进行特征工程](https://miro.medium.com/v2/resize:fit:560/1*KwpHmFl9gwhXAISYwNNaiw.png)

后面，我将更详细地介绍此应用程序。我将发布一篇有关图像数据特征工程的文章。除了上述方法外，它还将包括灰度、裁剪和边缘检测。

正好，我最近也在写关于 CV 方向的基础教程《[人工智能CV核心知识](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzA4NzE4MDQzMg==&action=getalbum&album_id=3503005363236880386&from_itemidx=1&from_msgid=2648753419#wechat_redirect)》，有兴趣的可以去查看下。

希望这篇文章对你有所帮助！你还可以阅读我的其他文章，或者查看有关企业 AI 实战项目的教程，相信会让你拥有更多收获。

**「AI秘籍」系列课程：**

[人工智能应用数学基础](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzA4NzE4MDQzMg==&action=getalbum&album_id=3074770001140400130&scene=173&subscene=227&sessionid=undefined&enterid=1717956640&from_msgid=2648748678&from_itemidx=1&count=3&nolastread=1#wechat_redirect)

[人工智能Python基础](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzA4NzE4MDQzMg==&action=getalbum&album_id=3035995870421073928&scene=126&uin=&key=&devicetype=iMac+MacBookPro17%2C1+OSX+OSX+14.5+build\(23F79\)&version=13080710&lang=zh_CN&nettype=WIFI&ascene=78&fontScale=100)

[人工智能基础核心知识](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzA4NzE4MDQzMg==&action=getalbum&album_id=3123885735829061633&scene=173&subscene=227&sessionid=1717956706&enterid=1717956713&from_msgid=2648751062&from_itemidx=1&count=3&nolastread=1#wechat_redirect)

[人工智能BI核心知识](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzA4NzE4MDQzMg%3D%3D&action=getalbum&album_id=3193965113967132673&subscene=&sessionid=svr_548800ce471&enterid=1704001437&from_msgid=2648751621&from_itemidx=1&count=3&nolastread=1&scene=21#wechat_redirect)

[人工智能CV核心知识](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzA4NzE4MDQzMg==&action=getalbum&album_id=3503005363236880386&from_itemidx=1&from_msgid=2648753419#wechat_redirect)

![AI企业项目实战课优惠二维码](https://cdn.jsdelivr.net/gh/hivandu/notes/img/202407252125233.png)

## 参考

[^1]: Github, https://github.com/hivandu/public_articles/blob/main/src/image_tools/colour_picker.ipynb