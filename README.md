# Titanic - Machine Learning from Disaster


```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```


```python
train = pd.read_csv("/kaggle/input/titanic/train.csv")
train.head()
```


```python
test = pd.read_csv("/kaggle/input/titanic/test.csv")
test.head()
```

Obviously, no 'Survived' column

### Encode categorical Variables


```python
df = pd.concat([train, test], axis=0, sort=True)

#Convert to category dtype
df['Sex'] = df['Sex'].astype('category')
df['Sex'] = df['Sex'].cat.codes
df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)
del df['Embarked']
df.drop(['Name', 'Cabin', 'Ticket', 'PassengerId'], axis=1, inplace=True)
```


```python
df.head()
```

### Scale continuous variables


```python

```
