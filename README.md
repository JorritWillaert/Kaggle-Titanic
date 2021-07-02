# Titanic - Machine Learning from Disaster


```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as functional
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

    /kaggle/input/titanic/train.csv
    /kaggle/input/titanic/test.csv
    /kaggle/input/titanic/gender_submission.csv
    


```python
train = pd.read_csv("/kaggle/input/titanic/train.csv")
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
test = pd.read_csv("/kaggle/input/titanic/test.csv")
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



Obviously, no 'Survived' column


```python
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
```

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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>7.2500</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>71.2833</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>7.9250</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.0</td>
      <td>53.1000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>8.0500</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Scale continuous variables


```python
continuous = ['Age', 'Fare', 'Parch', 'Pclass', 'SibSp']

scaler = StandardScaler()

for var in continuous:
    df[var] = df[var].astype('float32')
    df[var] = scaler.fit_transform(df[var].values.reshape(-1, 1))
```


```python
display_all(df.describe(include='all').T)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>1046.0</td>
      <td>-9.801147e-09</td>
      <td>1.000478</td>
      <td>-2.062328</td>
      <td>-0.616463</td>
      <td>-0.130575</td>
      <td>0.632964</td>
      <td>3.478882</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>1308.0</td>
      <td>0.000000e+00</td>
      <td>1.000382</td>
      <td>-0.643529</td>
      <td>-0.490921</td>
      <td>-0.364161</td>
      <td>-0.039051</td>
      <td>9.258680</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>1309.0</td>
      <td>1.602814e-08</td>
      <td>1.000382</td>
      <td>-0.444999</td>
      <td>-0.444999</td>
      <td>-0.444999</td>
      <td>-0.444999</td>
      <td>9.956863</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>1309.0</td>
      <td>2.185655e-08</td>
      <td>1.000382</td>
      <td>-1.546098</td>
      <td>-0.352091</td>
      <td>0.841916</td>
      <td>0.841916</td>
      <td>0.841916</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>1309.0</td>
      <td>6.440031e-01</td>
      <td>0.478997</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>1309.0</td>
      <td>-3.497049e-08</td>
      <td>1.000382</td>
      <td>-0.479087</td>
      <td>-0.479087</td>
      <td>-0.479087</td>
      <td>0.481288</td>
      <td>7.203909</td>
    </tr>
    <tr>
      <th>Survived</th>
      <td>891.0</td>
      <td>3.838384e-01</td>
      <td>0.486592</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Embarked_C</th>
      <td>1309.0</td>
      <td>2.062643e-01</td>
      <td>0.404777</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Embarked_Q</th>
      <td>1309.0</td>
      <td>9.396486e-02</td>
      <td>0.291891</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Embarked_S</th>
      <td>1309.0</td>
      <td>6.982429e-01</td>
      <td>0.459196</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


### Neural Network

Seperate back *train* and *test* data


```python
X_train = df[pd.notnull(df['Survived'])].drop(['Survived'], axis=1)
y_train = df[pd.notnull(df['Survived'])]['Survived']
X_test = df[pd.notnull(df['Survived'])].drop(['Survived'], axis=1)
```

Convert to tensors


```python
X_train_t = torch.tensor(X_train.values)
y_train_t = torch.tensor(y_train.values).float()
X_test_t = torch.tensor(X_train.values)
```

Create a *cross validation* set


```python
train_test_size = 0.8

seed_random = 123
X_train_t, X_val_t = train_test_split(X_train_t, random_state = seed_random, train_size = train_test_size, shuffle = True)
y_train_t, y_val_t = train_test_split(y_train_t, random_state = seed_random, train_size = train_test_size, shuffle = True)
```


```python
print(X_train_t.shape, X_val_t.shape)
print(y_train_t.shape, y_val_t.shape)
print(X_test_t.shape)
```

    torch.Size([712, 9]) torch.Size([179, 9])
    torch.Size([712]) torch.Size([179])
    torch.Size([891, 9])
    torch.float32 torch.float32
    


```python
class TitanicDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels.type(torch.LongTensor)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
```


```python
dataset_train = TitanicDataset(features = X_train_t, labels = y_train_t)
dataset_val = TitanicDataset(features = X_val_t, labels = y_val_t)

dataloader_train = DataLoader(dataset = dataset_train, batch_size = 64, shuffle = True)
dataloader_val = DataLoader(dataset = dataset_val, batch_size = 64, shuffle = True)
```


```python
class Network(nn.Module):
    def __init__(self, channels_in):
        super().__init__()
        
        self.fc1 = nn.Linear(channels_in, 250)
        self.fc2 = nn.Linear(250, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = functional.relu(x)
        x = self.fc2(x)
        x = functional.relu(x)
        x = functional.sigmoid(x)
        
        return x
```


```python
network = Network(X_train_t.shape[1])
```


```python
def train(network, device, dataloader_train, dataloader_val, loss_function, optimizer, epochs):
    network.to(device = device)
    train_loss, test_loss = [], []
    
    for epoch in range(epochs):
        print('Epoch: ' + str(epoch))
        for phase in ['train', 'val']:
            if phase == 'train':
                network.train(True)
                dataloader = dataloader_train
            else:
                network.train(False)
                dataloader = dataloader_val
            
            actual_loss = 0.0
            batch = 0
            
            for features, labels in dataloader:
                features = features.to(device = device)
                labels = labels.to(device = device)
                batch += 1
                
                if phase == 'train':
                    outputs = network(features)
                    loss = loss_function(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs = network(features)
                        loss = loss_function(outputs, labels)
                
                actual_loss += loss.item() * dataloader.batch_size
                
            epoch_loss = actual_loss / len(dataloader.dataset)
            print('Phase: ' + str(phase) + ', epoch loss: ' + str(epoch_loss))
            
            if phase == 'train':
                train_loss.append(epoch_loss)
            else:
                test_loss.append(epoch_loss)
    print('Training complete')
    return train_loss, test_loss
```


```python
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(network.parameters())
```


```python
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print('Training on device: ' + str(device))

train(network, device, dataloader_train, dataloader_val, loss_function, optimizer, epochs = 50)
```

    Training on device: cpu
    Epoch: 0
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 1
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 2
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 3
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 4
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 5
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 6
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 7
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 8
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 9
    

    /opt/conda/lib/python3.7/site-packages/torch/nn/functional.py:1639: UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
      warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
    

    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 10
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 11
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 12
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 13
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 14
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 15
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 16
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 17
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 18
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 19
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 20
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 21
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 22
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 23
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 24
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 25
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 26
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 27
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 28
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 29
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 30
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 31
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 32
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 33
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 34
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 35
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 36
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 37
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 38
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 39
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 40
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 41
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 42
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 43
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 44
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 45
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 46
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 47
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 48
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Epoch: 49
    Phase: train, epoch loss: nan
    Phase: val, epoch loss: nan
    Training complete
    




    ([nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan],
     [nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan,
      nan])




```python

```


```python

```


```python

```
