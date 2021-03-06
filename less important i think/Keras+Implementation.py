
# coding: utf-8

# In[19]:

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from numpy import*
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping


# In[20]:

# ## replacing NAN's with 0's

# np.random.seed(0)
# # load dataset
# dataset = np.genfromtxt("dataset.csv", delimiter=",")

# where_are_NaNs = isnan(dataset)
# dataset[where_are_NaNs] = 0

# X = dataset[:,2:12]
# Y = dataset[:,1]

# X_train, X_test, Y_train, Y_test= train_test_split(X, Y, random_state=4)

# X.shape


# In[21]:

# removed all the non numerical columns

raw_data = pd.read_csv('dataset.csv', sep=",")
df = raw_data.dropna(how='any') #to drop if any value in the row has a nan

Y = df['default'].as_matrix()

vardescr = pd.read_csv('variabledescr.csv')

# select variables with type=numeric
numerical_variables = vardescr[vardescr.Type == 'numeric'].Variable
X_df = df[numerical_variables]
X = X_df.as_matrix() #keras wants a matrix

X_train, X_test, Y_train, Y_test= train_test_split(X, Y, random_state=4)


scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
scaler = StandardScaler().fit(X_test)
X_test = scaler.transform(X_test)


# In[22]:

# create model
model = Sequential()
model.add(Dense(16, input_dim=25, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[30]:

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[31]:

# # Early stopping
# early_stopping_monitor = EarlyStopping(patience = 3)


# In[32]:

# Fit the model
model.fit(X_train, Y_train, epochs=50, batch_size=64)


# In[33]:

model.summary()


# In[34]:

# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:



