import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('creditcard.csv')
print(df.head())
print(df.isnull().sum())
print(df.describe().transpose())
print(df.info())

#Data Visualization
plt.figure(figsize=(12,8))
sns.histplot(df['Amount'], bins=50, kde=False)
plt.show()

sns.displot(df['Class'],bins = 30, kde = False)
plt.show()

plt.figure(figsize=(15, 8))  # adjust size for readability
corr = df.corr(numeric_only=True)

sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis",
            cbar=True, square=True)

plt.title("Correlation Heatmap of Credit Card Dataset", fontsize=14)
plt.show()


#Data Preprocessing
from sklearn.model_selection import train_test_split
df = df.drop('Time',axis=1)

X = df.drop('Class', axis=1).values
Y = df['Class'].values

x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.3, random_state=101)

#Data Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Balancing the weight
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

#Model Creation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)


model = Sequential()
model.add(Dense(78, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))  

model.compile(loss='binary_crossentropy', optimizer='adam')

#Model Training
model.fit(
    x_train,
    y_train,
    epochs=300,
    batch_size=256,
    validation_data=(x_test, y_test),
    callbacks=[early_stop],
    class_weight = class_weights
)

#Loss Visualization
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
plt.show()

#Evaluation and Prediction of Model
from sklearn.metrics import classification_report, confusion_matrix



predictions = (model.predict(x_test) > 0.5).astype("int32")

from sklearn.metrics import roc_auc_score, roc_curve
auc = roc_auc_score(y_test, predictions)
print("AUC:", auc)


print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))

# ROC Visualization
fpr, tpr, _ = roc_curve(y_test, predictions)
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


#Using a random to test the model
import random
random.seed(101)
random_ind = random.randint(0,len(df))

new_card = df.drop('Class', axis=1).iloc[random_ind]
new_card = new_card.values.reshape(1, -1)  
new_card = scaler.transform(new_card)
print(model.predict(new_card))
