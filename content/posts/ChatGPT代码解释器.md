---
title: "ChatGPT 代码解释器：如何为我节省数小时的工作"
date: 2023-07-10T01:22:07+08:00
draft: false
tags: []
summary: "**创建一个交互式世界地图，显示国家人口数量，配以简短的句子描述。**"
---

![img](https://qiniu.hivan.me/picGo/20230710011251.jpeg?imgNote)

2023 年 7 月 6 日，OpenAI 宣布 Code Interpreter 将在接下来的一周内向 ChatGPT Plus 用户开放。它可能是增强 ChatGPT 的能力和功能的最佳插件之一。

Code Interpreter 可以运行代码，允许上传数据，这样您就可以用它来进行数据清理、分析、可视化等许多其他任务。它就像是您指尖上的数据分析师。

听起来很棒吧？接下来我们来看看。

我在一项任务中使用了它，这项任务可能需要我花费几个小时才能完成。根据您的任务和对 Python 库的熟悉程度，这项任务甚至可能需要更长时间。

在使用 ChatGPT 时，我只需要写几句话并上传两个 CSV 文件。

我将逐步介绍整个过程，以及 Code Interpreter 在理解数据、清理和预处理数据以及创建数据可视化方面的印象深刻之处。

## **您需要先启用它**

如果您想在聊天中使用代码解释器，需要先通过设置启用它。

![如何启用代码解释器](https://qiniu.hivan.me/picGo/20230710011257.png?imgNote)

如何启用代码解释器

开启之后，你在接下来开启新的聊天的时候就可以使用它了。

## 来个具体任务

我打算创建一个世界地图可视化，其中点的大小与国家的人口成比例。

我有两个数据集，一个是国家人口，另一个是以纬度和经度表示的位置。

您可以轻松地在网上找到这些数据集。如果没有，文末我将提供数据集（以及源代码）。

## 让我们继续吧

由于这是一个简单的任务，我没有花费太多时间在提示工程上。这是我向 ChatGPT 请求的第一句话：

![img](https://qiniu.hivan.me/picGo/20230710011324.png?imgNote)

你可以在提示窗口左侧使用加号上传数据集。

![使用+号上传数据集](https://qiniu.hivan.me/picGo/20230710011303.png?imgNote)

使用+号上传数据集

首先要做的是读取数据集，这是显而易见的事情。但这还不够令人印象深刻！

```python
# Code Interpreter 生成的代码
import pandas as pd

# 将 CSV 数据加载到 pandas DataFrame 中
population_data = pd.read_csv('/mnt/data/country_population_historic.csv')

# 显示 DataFrame 的前几行
population_data.head()
```

然后，ChatGPT 要求提供国家位置：

![img](https://qiniu.hivan.me/picGo/20230710011345.png?imgNote)

由于我已经拥有相关数据了，所以我要做的就只是上传它：

![img](https://qiniu.hivan.me/picGo/20230710011400.png?imgNote)

```
# Code Interpreter 生成的代码
# 加载国家坐标数据
country_coordinates = pd.read_csv('/mnt/data/country_coordinates.csv')

# 显示前几行
country_coordinates.head()
```

## **数据清洗和预处理**

读取数据后的下一步是清洗和预处理数据。我相信你已经听过很多次，这一步通常需要最多的时间。

在创建世界地图可视化之前，有一些问题需要处理。ChatGPT 检测到它们并在不到一分钟的时间内提供了解决方案。相当令人印象深刻！

![img](https://qiniu.hivan.me/picGo/20230710011419.png?imgNote)

上面代码段中的代码:

```python
# 由代码解释器生成的代码
# 检查数据框中的国家名称是否一致
population_countries = set(population_data['Country Name'])
coordinates_countries = set(country_coordinates['name'])

# 人口数据中的国家但不在坐标数据中
missing_in_coordinates = population_countries - coordinates_countries

# 坐标数据中的国家但人口数据中没有
missing_in_population = coordinates_countries - population_countries
```

在坐标数据集和人口数据集中都有一些国家缺失。

ChatGPT 在上面的代码片段中做了什么：

- 创建了 Python 集合，包含两个数据集中的国家名称。
- 分别比较这些集合，找到它们之间的差异。

ChatGPT 在不到一分钟的时间内解决了这个问题：

![img](https://qiniu.hivan.me/picGo/20230710011441.png?imgNote)

以上片段中的代码：

```
# Code Interpreter 生成的代码
# 将人口数据集中的国家名称映射到坐标数据集中
country_name_mapping = {
    'Bahamas, The': 'Bahamas',
    'Brunei Darussalam': 'Brunei',
    'Cabo Verde': 'Cape Verde',
    'Congo, Dem. Rep.': 'Congo [DRC]',
    'Congo, Rep.': 'Congo [Republic]',
    "Cote d'Ivoire": "Côte d'Ivoire",
    'Czechia': 'Czech Republic',
    'Egypt, Arab Rep.': 'Egypt',
    'Gambia, The': 'Gambia',
    'Hong Kong SAR, China': 'Hong Kong',
    'Iran, Islamic Rep.': 'Iran',
    "Korea, Dem. People's Rep.": 'North Korea',
    'Korea, Rep.': 'South Korea',
    'Kyrgyz Republic': 'Kyrgyzstan',
    'Lao PDR': 'Laos',
    'Macao SAR, China': 'Macau',
    'Micronesia, Fed. Sts.': 'Micronesia',
    'Myanmar': 'Myanmar [Burma]',
    'North Macedonia': 'Macedonia [FYROM]',
    'Russian Federation': 'Russia',
    'Sao Tome and Principe': 'São Tomé and Príncipe',
    'Slovak Republic': 'Slovakia',
    'Syrian Arab Republic': 'Syria',
    'Turkiye': 'Turkey',
    'Venezuela, RB': 'Venezuela',
    'Yemen, Rep.': 'Yemen',
    'Eswatini': 'Swaziland'
}

# 清理人口数据集
population_data_clean = population_data.replace({"Country Name": country_name_mapping})

# 删除在坐标数据集中不存在的行
population_data_clean = population_data_clean[population_data_clean['Country Name'].isin(coordinates_countries)]

# 将人口数据与国家坐标数据合并
merged_data = pd.merge(population_data_clean, country_coordinates, left_on='Country Name', right_on='name')

# 只保留必要的列
merged_data = merged_data[['Country Name', '2022', 'latitude', 'longitude']]

# 重命名列以增加清晰度
merged_data.columns = ['Country', 'Population', 'Latitude', 'Longitude']

merged_data.head()
```

![img](https://qiniu.hivan.me/picGo/20230710011502.png?imgNote)

清理和预处理的数据

现在我们已经有了整洁和清晰的数据格式。这个步骤通常需要很多时间和人工。ChatGPT 在大约一分钟内完成了这一步。

## 创建可视化

ChatGPT 首先尝试使用 Matplotlib 库进行创建。然而，由于没有世界地图图像而失败：

![img](https://qiniu.hivan.me/picGo/20230710011516.png?imgNote)

然后，我只是要求它使用 Plotly，这就是我一开始想到的。

![img](https://qiniu.hivan.me/picGo/20230710011535.png?imgNote)

![img](https://qiniu.hivan.me/picGo/20230710011549.png?imgNote)

以上代码段中的代码：

```
# 代码由代码解释器生成
import plotly.graph_objects as go

# 创建散点地理图
fig = go.Figure(data=go.Scattergeo(
    lon = merged_data['Longitude'],
    lat = merged_data['Latitude'],
    text = merged_data['Country'] + ': ' + merged_data['Population'].astype(str),
    mode = 'markers',
    marker = dict(
        size = (merged_data['Population'] / 1000000) ** 0.5,  # Take square root to better visualize differences
        sizemode = 'diameter',
        sizeref = 1,
        sizemin = 4,
        color = merged_data['Population'],
        colorscale = 'Viridis',
        colorbar_title = "Millions",
    )
))

# 更新布局
fig.update_layout(
    title_text = '2022 World Population',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection_type = 'equirectangular'
    )
)

fig.show()
```

它不能在浏览器中显示交互式世界地图，但成功编写了生成它的代码。我所要做的就是将代码复制粘贴到 Jupyter 笔记本中。

这是最终结果：

![img](https://www.notion.so/image/https%3A%2F%2Fmiro.medium.com%2Fv2%2Fresize%3Afit%3A700%2F1*muaYWUJQNpEt0Tpm_UaHzw.gif?id=20caa395-3ab2-4b2b-8a15-d6c4fe9a1ebf&table=block&spaceId=9a074068-2ca2-46e6-b231-935d54db43cd&userId=4f93353d-181a-4410-94d6-f9f4b3a98955&cache=v2)

互动世界地图，显示国家人口

## **最后的话**

我们所做的事情：

- 读取数据集
- 清洗、预处理和合并它们
- 创建互动数据可视化

我们所要做的只是写两个句子（并告诉 ChatGPT 使用 Plotly）。我认为这非常令人印象深刻！

## 数据

本文已经结束。

文章最后，我将提供数据以及一个 jupyter notebook 内容，和往常一样，数据将付费查看，以获取一些成本。有想要的朋友可以去我公众号内搜索本文购买：

![个人公众号：坍缩的奇点](https://qiniu.hivan.me/picGo/20230704000058.png?imgNote)