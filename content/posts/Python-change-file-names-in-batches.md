---
title: "Python 批量修改文件名"
date: 2023-10-17T17:44:13+08:00
draft: false
tags: []
summary: "> 仅个人需求，有需要的可以自取。

前段时间为家里孩子下载了一批课程，但是文件命名就很奇怪也很乱，就想着将文件名修改掉便于查看。"
---

这批视频下载下来后前边都给了诸如`001`之类的编号，当然是序列。可是这批序列又非常的乱，比如，「数列」和「导数」给的是考前的编号，而课本上要先学习的「集合」，「逻辑」，「不等式」等又编号又很靠后。不仅如此，就算是同一部分，其中的编号也是混乱的。



![image-20231017172008515](https://raw.githubusercontent.com/hivandu/notes/main/img/20231017173523.png)

那么就有了批量修改文件名的需求，当然我第一时间想到的是`Better rename`，已经是一个很古老的版本了：

![](https://files.mdnice.com/user/43981/baf01bcb-fdf3-4150-97df-b99ed3b4a375.png)



可是当我使用的时候才发现并不能满足我的个人需求，也许是我不太会用吧。起码，我是想删掉开头的那些序列以及其中重复不必要的内容。但是这玩意并不能支持正则或者相关的功能。没办法，眼见有几种方式去做，一种是 Mac 自带的「自动操作」，一种是「捷径」，还有就是干脆用 Python 写个脚本。

所以，我使用了自己觉得最简便的方式，写了这样一个脚本：

```python
import os
import tkinter as tk
from tkinter import filedialog
import re

root = tk.Tk()
root.withdraw()

folderPath = filedialog.askdirectory() # 获得选择好的文件夹
# filePath = filedialog.askopenfilename() # 获得选择好的文件

print(folderPath)
# print(filePath)

files = os.listdir(folderPath)

fileList = []
newFileList = []

oldStr = input('请输入要修改的内容或正则:')
newStr = input('请输入要替换的内容，不修改只删除可留空:')

for i in files:
    if i[0] != '.':
        portion = os.path.splitext(i)
        newname = re.sub(r'{}'.format(oldStr), r'{}'.format(newStr), portion[0])
        # os.chdir(folderPath) # 测试完毕后要正式修改文件名取消这里注释
        # os.rename(portion[0]+portion[1], newname+portion[1]) # 测试完毕后要正式修改文件名取消这里注释
        
        fileList.append(portion[0]+portion[1])
        newFileList.append(newname+portion[1])

print('修改前，一共有{}个文件\n  {}'.format(len(fileList), fileList))
print('修改后，一共有{}个文件\n  {}'.format(len(newFileList), newFileList))
```

有需要的小伙伴可以自取了。代码执行后会让你选取你要修改的文件的目录，然后会让你输入你要修改的内容，可以是正则，然后输入你要修改成的内容。

![image-20231017173302128](https://raw.githubusercontent.com/hivandu/notes/main/img/20231017173524.png)

比如，我需要修改标题：

```
# 输入需要替换的内容
\d{3} - 
```



![image-20231017173338085](https://raw.githubusercontent.com/hivandu/notes/main/img/20231017173525.png)

要替换的内容我直接留空回车，打印结果：

```python
修改前，一共有 16 个文件
  ['001 - 不等式 1.1 不等式的基本性质 高中数学.mp4', '016 - 不等式 4.4 恒成立与存在性问题 高中数学.mp4', '007 - 不等式 3.3 高次不等式 高中数学.mp4', '008 - 不等式 3.4 含参讨论 高中数学.mp4', '003 - 不等式 1.3 比大小思路 高中数学.mp4', '002 - 不等式 1.2 不等式证明思路 高中数学.mp4', '006 - 不等式 3.1 二次不等式 高中数学.mp4', '015 - 不等式 1.4 复杂证明题 高中数学.mp4', '005 - 不等式 2.3 基本不等式变式 高中数学.mp4', '011 - 不等式 2.4 不要过度放缩三元不等式 高中数学.mp4', '010 - 不等式 2.2 使用条件误区 高中数学.mp4', '009 - 不等式 4.1 对勾型函数最值 高中数学.mp4', '013 - 不等式 4.2 对称与均值 高中数学.mp4', '004 - 不等式 2.1 基本不等式均值不等式 高中数学.mp4', '014 - 不等式 4.3 齐次与均值 高中数学.mp4', '012 - 不等式 3.2 穿针引线法 高中数学.mp4']
修改后，一共有 16 个文件
  ['不等式 1.1 不等式的基本性质 高中数学.mp4', '不等式 4.4 恒成立与存在性问题 高中数学.mp4', '不等式 3.3 高次不等式 高中数学.mp4', '不等式 3.4 含参讨论 高中数学.mp4', '不等式 1.3 比大小思路 高中数学.mp4', '不等式 1.2 不等式证明思路 高中数学.mp4', '不等式 3.1 二次不等式 高中数学.mp4', '不等式 1.4 复杂证明题 高中数学.mp4', '不等式 2.3 基本不等式变式 高中数学.mp4', '不等式 2.4 不要过度放缩三元不等式 高中数学.mp4', '不等式 2.2 使用条件误区 高中数学.mp4', '不等式 4.1 对勾型函数最值 高中数学.mp4', '不等式 4.2 对称与均值 高中数学.mp4', '不等式 2.1 基本不等式均值不等式 高中数学.mp4', '不等式 4.3 齐次与均值 高中数学.mp4', '不等式 3.2 穿针引线法 高中数学.mp4']
```

打印内容中查看自己修改前和修改后的文件对比，感觉没问题了，把其中注释的两行代码打开注释，就可以完成文件修改了。



![image-20231017173440797](https://raw.githubusercontent.com/hivandu/notes/main/img/20231017173526.png)