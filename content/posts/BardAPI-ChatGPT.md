---
title: "将 Bard API 与 ChatGPT 集成：实时数据访问"
date: 2023-07-21T12:47:54+08:00
draft: false
tags: ["AI"]
summary: "在人工智能领域，很少有创新能像 OpenAI 的 ChatGPT 一样激发世界的想象力。这种非凡的对话式人工智能改变了我们看待人机交互的方式，展现出一定程度的复杂性、情境意识和创造力，而这些曾经被认为是人类智能的专属领域。"
---

![img](https://miro.medium.com/v2/resize:fit:700/1*sBgoRLJEfkoRRl3BYrBV2g.png)

ChatGPT 基于强大的 GPT-3 模型构建，能够进行引人入胜、有意义且令人印象深刻的类人对话。它可以写诗、回答复杂的问题、辅导各种科目、翻译语言，甚至模仿著名作家的写作风格。从本质上讲，它重新定义了我们认为人工智能可能实现的界限。

然而，ChatGPT 的主要缺点是它缺乏实时互联网数据访问。这意味着，虽然 ChatGPT 可以生成高度智能且上下文准确的响应，但其知识基本上被及时冻结，截止日期为 2021 年 9 月。

那么，当出现需要通过将 Google 的 Bard API 与 ChatGPT 集成来获取超出此限制的信息的问题时，会发生什么情况呢？

**以下是使用 Python**将 Bard API 连接到 ChatGPT 以检索实时数据的分步指南：

## 第 1 步：安装非官方 Bard Python 库并检索 Cookie 值（API 密钥）

我正在使用[Daniel Park](https://github.com/dsdanielpark/Bard-API)使用逆向工程开发的非官方 Bard 库。这个库是一个非常用户友好的 Python 包。其主要目的是通过 API 从 Google Bard 获取响应。使用 Bard-API，用户可以方便地将 Bard 的自然语言响应集成到他们的 Python 项目和各种应用程序中。

```
pip install bardapi
```

您还可以直接从 Github 安装[最新版本：](https://github.com/dsdanielpark/Bard-API)

```
pip install git+https://github.com/dsdanielpark/Bard-API.git
```

```python
from bardapi import Bard

token = 'xxxxxxx'
bard = Bard(token=token)
bard.get_answer(<your query>)['content']
```



**设置您的 API 密钥**

安装 Bard-API 后，使用 Bard cookie 中的 Secure-1PSID 进行身份验证。尽管非正式地称为 API KEY（Cookie 值），但请记住对其保密以确保安全访问。

1. 访问 https://bard.google.com/

2. 按 F12 或右键单击并“检查”

   ![img](https://qiniu.hivan.me/picGo/20230721124228.png?imgNote)

3. 转到应用程序 → Cookie，并将您的 __Secure-1PSID Cookie 值复制到安全位置。

## [步骤 2：从 openai.com](http://openai.com/)获取 OpenAI 密钥并安装 OpenAI 库

![img](https://qiniu.hivan.me/picGo/20230721124257.png?imgNote)

访问 OpenAI 网站并获取您的 OpenAI API 密钥。现在安装 OpenAI 库并导入它。

```
pip install openai
```

```python
import openai
openai.api_key = <Your_API_Key>
```



## 步骤 3：将 bard 请求结果连接到 gpt-3.5-turbo 模型并设计提示

这里的关键部分是设计将 Bard 结果集成到 ChatGPT API 函数中所需的提示。因此，我为此制定了一个方法：

```python
query = input("Your query")
bard_result = bard.get_answer(query)['content']
completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                   {"role": "system", "content": "Act as an AI chatbot with access to the internet."},
                   {"role": "user", "content": "Provide a well structured and easily readable text by analyzing this: The first content below is the user's query and the second content below is the result obtained by accessing the internet with the help of google's search alogoritm. Provide the well structured and good mannered answer by processing the user's query and the result from Google search algorithm. /n"+query+' /n '+ bard_result}
       
                 ]
)
final_response = completion["choices"][0]["message"]["content"]
print(final_response)
```

将所有代码封装在一起，得出结果：

```Python
from bardapi import Bard
import openai
openai.api_key = <Your Key>
token = <Your Key>
bard = Bard(token=token)
query = input("Your query: ")
bard_result = bard.get_answer(query)['content']
completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                   {"role": "system", "content": "Act as an AI chatbot with access to the internet."},
                   {"role": "user", "content": "Provide a well structured and easily readable text by analyzing this: The first content below is the user's query and the second content below is the result obtained by accessing the internet with the help of google's search alogoritm. Provide the well structured and good mannered answer by processing the user's query and the result from Google search algorithm. /n"+query+' /n '+ bard_result}
       
                 ]
)
final_response = completion["choices"][0]["message"]["content"]
print(final_response)
```

![img](https://qiniu.hivan.me/picGo/20230721124414.png?imgNote)

与其使用 gpt-3.5-turbo 型号，不如试试 gpt-3.5-turbo-16k 和 gpt-4-0314，效果会更好。

通过整合像谷歌的 Bard 这样的应用程序接口，ChatGPT 可以超越目前的局限，为用户提供实时、准确的上下文信息。这将大大增强其协助、教育和与用户互动的能力，为人与人工智能的互动增添一个全新的维度。此外，这还将极大地扩展 ChatGPT 的应用范围，为企业、教育工作者、研究人员和个人带来新的机遇。

我认为这是将互联网接入集成到 ChatGPT 并从 ChatGPT 获得实时见解的最简单方法。