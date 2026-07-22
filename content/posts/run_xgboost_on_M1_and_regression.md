---
title: "Run xgboost on Mac and Regression data"
date: 2024-01-01T00:00:00+08:00
draft: false
summary: "> The source code: [xgboost_regression](https://github.com/hivandu/practise/blob/master/improvement/xgboost_regression.ipynb)"
---

> update at 2021-09-07:

## Install xgboost on Apple M1

```
git clone --recursive https://github.com/dmlc/xgboost
mkdir xgboost/my_build
cd xgboost/my_build
CC=gcc-11 CXX=g++-11 cmake ..
make -j4
cd ../python_package
/Users/xx/miniforge3/envs/tf/bin/python setup.py install
```

> u must install miniforge for M1, `conda create -n tf python=3.9.5`

## Run xgboost

In the process of using xgboost, I encountered a small obstacle, that is, xgboost cannot be run normally on the M1 of the Mac. It needs to be tossed. The following is the installation process:

### 1. Homebrrew is required first

### 2. Install gcc and cmake

```shell
brew install gcc
brew install cmake
brew install libomp
```

### 3. Download xgboost package

Yes, you cannot use the network package to install, you need to download, compile and install by yourself. Fortunately, the process is not troublesome:

Source: [http://mirrors.aliyun.com/pypi/simple/xgboost/](http://mirrors.aliyun.com/pypi/simple/xgboost/)

I downloaded `xgboost-1.4.2.tar.gz`

### 4. Installation

Enter `cd ~/download/`

run

```shell
pip install xgboost-1.4.2.tar.gz
```

Okay, you can introduce it to try

```python
from xgboost import XGBClassifier
xb = XGBClassifier()
```

## xgboost Regression


### load data

```python
import pandas as pd
import warnings
%pylab inline

warnings.filterwarnings('ignore')

# load data from url
df = pd.read_csv('./data/Titanic.txt', sep=',', quotechar='"', encoding='ISO 8859-15')
df.info()
df.head()

# Filter some features
features = df[['pclass', 'age', 'sex']]
# Label
label = df['survived']
features.info()

# Missing values ​​are filled with mean
features['age'].fillna(df['age'].mean(), inplace=True)
features.info()

# Divide the dataset
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(features, label, test_size = 0.25, random_state=33)

# Feature vectorization
from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer(sparse = False)
train_x = vec.fit_transform(train_x.to_dict(orient='record'))
test_x = vec.transform(test_x.to_dict(orient='record'))

# Random forest training and prediction
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(train_x, train_y)
print('The accuracy of random Forest Classifier on testing set:', rfc.score(test_x, test_y))

"""
The accuracy of random Forest Classifier on testing set: 0.7781155015197568
"""


# xgboost training and prediction
from xgboost import XGBClassifier
xb = XGBClassifier()
xb.fit(train_x, train_y)
print(f'The accuracy:', xb.score(test_x, test_y))

"""
The accuracy: 0.7750759878419453
"""
```