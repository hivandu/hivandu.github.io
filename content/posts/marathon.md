---
title: "马拉松跑步数据"
date: 2021-09-23T11:33:29+08:00
draft: false
tags: []
summary: "先引入数据,准备进行分析"
---

```python
# 引入数据
import pandas as pd
data = pd.read_csv('~/data/cbcpv/marathon/marathon.csv')
data.sample(5)
```

**OUT:**

|       |  age | gender |      split |      final |
| ----: | ---: | -----: | ---------: | ---------: |
| 19841 |   34 |      M | `01:55:25` | `04:50:03` |
| 11002 |   28 |      W | `01:55:00` | `04:11:00` |
| 11619 |   26 |      M | `01:40:28` | `04:13:52` |
|  4068 |   34 |      M | `01:38:30` | `03:30:21` |
|  6922 |   35 |      M | `01:37:44` | `03:48:37` |



这个数据集有以下几个特征：

- age，运动员的年龄
- gender，运动员的性别
- split，半程所用时间
- final，全程所用时间，即最终成绩



自然,要先了解下数据的具体情况

```python
data.info()

# 输出结果
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 37250 entries, 0 to 37249
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   age     37250 non-null  int64 
 1   gender  37250 non-null  object
 2   split   37250 non-null  object
 3   final   37250 non-null  object
dtypes: int64(1), object(3)
memory usage: 1.1+ MB
```



可以看到并没缺失值, 不过`split`和`final`特征中的数据不实数字类型, 用字符串表示了所用时间长度, 所以我们需要进行转化:

```python
import datetime
def convert_time(s):
    h,m,s=map(int, s.split(':'))
    return datetime.timedelta(hours=h, minutes=m, seconds=s)
```



使用完成的方法进行数据转换,我们从新读取一下数据:

```python
df=pd.read_csv(
    '~/data/cbcpv/marathon/marathon.csv',
    converters={
        'split': convert_time,
        'final': convert_time
    }
)
df.dtypes

# 输出结果
age                 int64
gender             object
split     timedelta64[ns]
final     timedelta64[ns]
dtype: object
```



这次数据已经转换为`timedelta64`类型, 下面我们就要转化时间为整数, 一般的做法都是秒或者毫秒数:

```python
d = datetime.timedelta(hours=1, minutes=0, seconds=0)
df2 = pd.DataFrame({'time':[d]})
df2.astype(int)

# 输出结果
             time
0    3600000000000
```

我们看到的输出结果,是“纳秒“(ns)单位:

$$1s=10^9ns$$



我们还需要转化为秒:

```python
d=datetime.timedelta(hours=1, minutes=0, seconds=0)
df2=pd.DataFrame({'time':[d]})
df2.astype(int) * 1e-9

# out
       time
0    3600.0
```



然后我们要讲`split`和`final`两个特征的数据进行转化

```python
df['split_sec']=df['split'].astype(int) * 1e-9
df['final_sec']=df['final'].astype(int) * 1e-9
df.sample(5)
```

**OUT:**

|       |  age | gender |             split |             final | split_sec | final_sec |
| ----: | ---: | -----: | ----------------: | ----------------: | --------: | --------- |
| 11725 |   35 |      M | `0 days 01:53:53` | `0 days 04:14:19` |    6833.0 | 15259.0   |
| 19815 |   24 |      M | `0 days 01:58:45` | `0 days 04:49:57` |    7125.0 | 17397.0   |
|  5754 |   49 |      M | `0 days 01:42:39` | `0 days 03:41:05` |    6159.0 | 13265.0   |
| 33166 |   46 |      M | `0 days 02:31:37` | `0 days 06:06:17` |    9097.0 | 21977.0   |
|  9226 |   36 |      W | `0 days 01:49:06` | `0 days 04:01:55` |    6546.0 | 14515.0   |



现在多了两个特征`split_sec`和`final_sec`,  都是以秒为单位的浮点数.



## 描述统计

先了解数据:

```python
df.describe()
```

**OUT:**

|       |          age |                       split |                       final |    split_sec | final_sec    |
| ----: | -----------: | --------------------------: | --------------------------: | -----------: | ------------ |
| count | 37250.000000 |                       37250 |                       37250 | 37250.000000 | 37250.000000 |
|  mean |    40.697369 | `0 days 02:03:54.425664429` | `0 days 04:48:09.303597315` |  7434.425664 | 17289.303597 |
|   std |    10.220043 | `0 days 00:22:55.093889674` | `0 days 01:03:32.145345151` |  1375.093890 | 3812.145345  |
|   min |    17.000000 |           `0 days 01:05:21` |           `0 days 02:08:51` |  3921.000000 | 7731.000000  |
|   25% |    33.000000 |           `0 days 01:48:25` |           `0 days 04:02:24` |  6505.000000 | 14544.000000 |
|   50% |    40.000000 |           `0 days 02:01:13` |           `0 days 04:44:25` |  7273.000000 | 17065.000000 |
|   75% |    48.000000 |           `0 days 02:16:11` |           `0 days 05:27:36` |  8171.000000 | 19656.000000 |
|   max |    86.000000 |           `0 days 04:59:49` |           `0 days 10:01:08` | 17989.000000 | 36068.000000 |

居然年龄上最大的数据是 86,  让我们看看特征的数据分布:

```python
%matplotlib inline
import seaborn as sns
ax=sns.boxplot(x=df['age'])
```

![image-20210923215638259](https://qiniu.hivan.me/picGo/20210923215638.png?imgNote)

这个箱线图反应了, 数据里确实有一些“离群值”.

## 数据分布

研究下数据分布, 看看`split_sec`和`final_sec`

```python
sns.displot(df['split_sec'])
```

![image-20210923215655517](https://qiniu.hivan.me/picGo/20210923215655.png?imgNote)

```python
sns.displot(df['final_sec'])
```

![image-20210923215712676](https://qiniu.hivan.me/picGo/20210923215712.png?imgNote)

整体看来,两个特征下的数据都符合正态分布, 但是`final_sec`的分布图比较胖.



这次我们把`gender`这个分类特征添加进来:

```python
sns.violinplot(x='gender', y='final_sec', data=df)
```

![image-20210923215729962](https://qiniu.hivan.me/picGo/20210923215730.png?imgNote)

这些看到, 男性运动员在总体上还是比女性运动员要快一些.



## 寻找优秀的原因

跑马拉松或者了解这项运动的人都清楚, 运动员很关注整个赛程中前后半程的时间比较,好的选手是后半程用时和前半程近似. 因此, 我们来研究下, 这些运动员前后半程用时情况.

```python
g=sns.jointplot('split_sec', 'final_sec', data=df, kind='hex')

# 绘制一条直线, 作为参考
import numpy as np
g.ax_joint.plot(np.linspace(4000, 16000), np.linspace(8000, 32000), ':k')
```

![image-20210923215745469](https://qiniu.hivan.me/picGo/20210923215745.png?imgNote)

横坐标是`splict_sec`特征, 即半程用时. 纵轴表示`final_sec`特征, 全程用时. 途中可以看出, 的确是越优秀的运动员,前半程用时越接近全程用时的一半, 甚至还有少数后半程跑的更快的.

我们做个计算来深入研究下:

```python
df['split_frac']=1-2*df['split_sec']/df['final_sec']
df.sample(5)
```

**OUT:**

|       |  age | gender |             split |             final | split_sec | final_sec | split_frac |
| ----: | ---: | -----: | ----------------: | ----------------: | --------: | --------: | ---------- |
|  2065 |   35 |      W | `0 days 01:31:41` | `0 days 03:14:40` |    5501.0 |   11680.0 | 0.058048   |
|  9001 |   43 |      W | `0 days 01:58:19` | `0 days 04:00:44` |    7099.0 |   14444.0 | 0.017031   |
| 30039 |   34 |      M | `0 days 02:25:17` | `0 days 05:39:21` |    8717.0 |   20361.0 | 0.143755   |
| 27456 |   62 |      W | `0 days 02:13:28` | `0 days 05:25:01` |    8008.0 |   19501.0 | 0.178709   |
| 13335 |   41 |      M | `0 days 01:45:36` | `0 days 04:21:00` |    6336.0 |   15660.0 | 0.190805   |

用直方图再增加一个参考线来看看`split_frac`特征中的数据分布:

```python
import matplotlib.pyplot as plt
sns.displot(df['split_frac'], kde=False)
# 垂直于 x 轴的直线，0 表示 x 轴位置
plt.axvline(0, color='k', linestyle='--')
```

![image-20210923215830370](https://qiniu.hivan.me/picGo/20210923215830.png?imgNote)

从这张图中, 更清晰的看到全体参赛者的运动安排.

再来探究下不同特征之间的关系:

```python
sns.pairplot(
    data=df,
    vars=['age','split_sec','final_sec','split_frac'],
    hue='gender'
)
```

![image-20210923215848042](https://qiniu.hivan.me/picGo/20210923215848.png?imgNote)



让我们来看下 80 岁选手的数量:

```python
(df.age>=80).sum()

# OUT
15
```



下面, 我们划分下年龄段,看看各年龄段的成绩分布:

```python
df['age_dec']=df['age'].map(lambda age: 10*(age//10))
sns.violinplot(
    x='age_dec', 
    y='split_frac', 
    hue='gender', 
    data=df, 
    split=True, 
    inner='quartile', 
    palette=['lightblue', 'lightpink']
)
```

![image-20210923215931464](https://qiniu.hivan.me/picGo/20210923215931.png?imgNote)

看这张图, 我们发现,不同性别的运动员的`split_frac`特征数据分布中, 年龄越大,前后端的时间分布比相对集中.



再看看全程用时分布比较:

```python
sns.violinplot(
    x='age_dec',
    y='final_sec',
    hue='gender',
    data=df,
    split=True,
    inner='quartile',
    palette=['lightblue', 'lightpink']
)
```

![image-20210923215947281](https://qiniu.hivan.me/picGo/20210923215947.png?imgNote)

从 30 岁往后, 明显年纪越大,用时越长.