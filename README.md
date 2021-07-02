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
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)
df['Age'].fillna(df['Age'].median(), inplace = True)
df['Fare'].fillna(df['Fare'].median(), inplace = True)
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
      <td>1309.0</td>
      <td>2.914207e-09</td>
      <td>1.000382</td>
      <td>-2.273836</td>
      <td>-0.581628</td>
      <td>-0.116523</td>
      <td>0.426099</td>
      <td>3.914388</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>1309.0</td>
      <td>-2.914207e-09</td>
      <td>1.000382</td>
      <td>-0.643464</td>
      <td>-0.490805</td>
      <td>-0.364003</td>
      <td>-0.038786</td>
      <td>9.262028</td>
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
      <td>6.997708e-01</td>
      <td>0.458533</td>
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
    


```python
class TitanicDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels.type(torch.float32)
        self.labels = torch.unsqueeze(self.labels, 1)
        
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
        self.fc2 = nn.Linear(250, 1)
        
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
    train_loss, val_loss = [], []
    
    for epoch in range(epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                network.train(True)
                dataloader = dataloader_train
            else:
                network.train(False)
                dataloader = dataloader_val
            
            actual_loss = 0.0
            actual_acc = 0.0
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
                        acc = accuracy_metric(outputs, labels)
                        actual_acc += acc
                actual_loss += loss.item() * dataloader.batch_size
            
            actual_acc /= len(dataloader.dataset)
            epoch_loss = actual_loss / len(dataloader.dataset)
            if epoch % 10 == 0:
                print('Phase: ' + str(phase) + ', epoch loss: ' + str(epoch_loss))
                if phase == 'val':
                    print('Accuracy: ' + str(actual_acc))
                torch.save(network.state_dict(), "/kaggle/working/model_02_07_2021_epoch_" + str(epoch) + '.pt')
            
            if phase == 'train':
                train_loss.append(epoch_loss)
            else:
                val_loss.append(epoch_loss)
    print('Training complete')
    return train_loss, val_loss
```


```python
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(network.parameters(), lr = 0.0005)

def accuracy_metric(outputs, labels):
    acc = 0
    for i in range(outputs.shape[0]):
        if (outputs[i] > 0.5 and labels[i] == 1) or (outputs[i] <= 0.5 and labels[i] == 0):
            acc += 1
    return acc
```


```python
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print('Training on device: ' + str(device))

train_loss, val_loss = train(network, device, dataloader_train, dataloader_val, loss_function, optimizer, epochs = 250)
```

    Training on device: cpu
    Phase: train, epoch loss: 0.7370699764637465
    Phase: val, epoch loss: 0.721055590240649
    Accuracy: 0.6256983240223464
    Phase: train, epoch loss: 0.6766130200932535
    Phase: val, epoch loss: 0.6750174367894007
    Accuracy: 0.7821229050279329
    Phase: train, epoch loss: 0.6719620736797204
    Phase: val, epoch loss: 0.6604939892305343
    Accuracy: 0.7932960893854749
    Phase: train, epoch loss: 0.6471231278408779
    Phase: val, epoch loss: 0.6410782030840826
    Accuracy: 0.7988826815642458
    Phase: train, epoch loss: 0.6321745293863704
    Phase: val, epoch loss: 0.6386876026345365
    Accuracy: 0.8044692737430168
    Phase: train, epoch loss: 0.629747315738978
    Phase: val, epoch loss: 0.6383322603875698
    Accuracy: 0.8268156424581006
    Phase: train, epoch loss: 0.6311275128568157
    Phase: val, epoch loss: 0.6325195568233895
    Accuracy: 0.8379888268156425
    Phase: train, epoch loss: 0.617671629016319
    Phase: val, epoch loss: 0.6423453858444811
    Accuracy: 0.8212290502793296
    Phase: train, epoch loss: 0.6010052434514078
    Phase: val, epoch loss: 0.6348270224459345
    Accuracy: 0.8379888268156425
    Phase: train, epoch loss: 0.6272932438368208
    Phase: val, epoch loss: 0.6272513959660876
    Accuracy: 0.8324022346368715
    Phase: train, epoch loss: 0.614928583080849
    Phase: val, epoch loss: 0.6347299501216611
    Accuracy: 0.8324022346368715
    Phase: train, epoch loss: 0.6245186034213291
    Phase: val, epoch loss: 0.6367610526484484
    Accuracy: 0.8435754189944135
    Phase: train, epoch loss: 0.6123484493641371
    Phase: val, epoch loss: 0.6345060550966742
    Accuracy: 0.8379888268156425
    Phase: train, epoch loss: 0.6005783857924215
    Phase: val, epoch loss: 0.6418425277624716
    Accuracy: 0.8435754189944135
    Phase: train, epoch loss: 0.6222814817107125
    Phase: val, epoch loss: 0.6354557868488674
    Accuracy: 0.8435754189944135
    Phase: train, epoch loss: 0.5959445653336771
    Phase: val, epoch loss: 0.6377820915350035
    Accuracy: 0.8379888268156425
    Phase: train, epoch loss: 0.6120018637582157
    Phase: val, epoch loss: 0.6483729485026951
    Accuracy: 0.8435754189944135
    Phase: train, epoch loss: 0.6051727252060108
    Phase: val, epoch loss: 0.6477119829401624
    Accuracy: 0.8379888268156425
    Phase: train, epoch loss: 0.5927651678578237
    Phase: val, epoch loss: 0.6463567424752859
    Accuracy: 0.8379888268156425
    Phase: train, epoch loss: 0.5971969299102098
    Phase: val, epoch loss: 0.6489054077830394
    Accuracy: 0.8435754189944135
    Phase: train, epoch loss: 0.6096677244379279
    Phase: val, epoch loss: 0.6458713595427614
    Accuracy: 0.8491620111731844
    Phase: train, epoch loss: 0.6048277147700277
    Phase: val, epoch loss: 0.6477895768661073
    Accuracy: 0.8435754189944135
    Phase: train, epoch loss: 0.5955894529149773
    Phase: val, epoch loss: 0.644979679384711
    Accuracy: 0.8435754189944135
    Phase: train, epoch loss: 0.6013867024625286
    Phase: val, epoch loss: 0.644884652931597
    Accuracy: 0.8435754189944135
    Phase: train, epoch loss: 0.6025764915380585
    Phase: val, epoch loss: 0.6524175079175214
    Accuracy: 0.8435754189944135
    Training complete
    


```python

```


```python

```


```python

```
