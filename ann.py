# %% [markdown]
# # Import Pakeage

# %%
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist


# %% [markdown]
# # Data Preprocessing

# %%
# Load Fashion Minst Dataset 
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() 

# %%
# Data Reshape
x_train = x_train.reshape(60000, 784)
x_test_ori = x_test.copy()
x_test = x_test.reshape(10000, 784)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# %%
# Encording data
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# %%
# Data split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=264)
print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)

# %% [markdown]
# # Training ANN Model

# %%
# Model Structure
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dropout(0.3),

    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.3),

    BatchNormalization(),
    Dense(32, activation='relu'),
    Dropout(0.3),

    BatchNormalization(),
    Dense(24, activation='relu'),
    Dropout(0.3),

    BatchNormalization(),
    Dense(10,activation='sigmoid'),
])

model.summary()

# %%
model.compile(optimizer=Adam(learning_rate=0.1,decay=0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

# %%
history = model.fit(x_train, y_train,
                  batch_size=128,
                  epochs=160,
                  verbose=1,
                  validation_data=(x_val,y_val))

# %% [markdown]
# # Result

# %%
plt.figure(figsize=(10,3))
plt.subplot(121)
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.title('loss curve')
plt.ylabel('loss')
plt.legend()
plt.subplot(122)
plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_accuracy'],label='val_accuracy')
plt.title('accuracy curve')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# %%
result = model.evaluate(x_test,y_test)

# %%
print('MLP score: {:.2%}'.format(result[1]))

classes = ['T-shirt/Top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
          
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis = 1)
y_true = np.argmax(y_test, axis = 1)
confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

f,ax = plt.subplots(figsize = (12,9))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.1, fmt='.0f', ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=90)
plt.yticks(tick_marks, classes, rotation=0)
plt.title("Confusion Matrix")
plt.show()

# %%
# shirt:6 T-shirt:0
incorrect = []
for i in range(len(y_test)):
    if (y_pred_classes[i]==0) and (y_true[i]==6):
        incorrect.append(i)

print('Predict shirt as T-shirt:',len(incorrect))

fig, ax = plt.subplots(1, 5, figsize=(12, 3))
# Plot a boxplot with Seaborn
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(x_test_ori[incorrect[i]], cmap='binary')
    plt.title(incorrect[i])
fig.suptitle('Wrong Predictions: Presition=T-shirt/Label=Shirt')
plt.show()


