# %% [code] {"execution":{"iopub.status.busy":"2021-09-21T21:20:03.38408Z","iopub.execute_input":"2021-09-21T21:20:03.384652Z","iopub.status.idle":"2021-09-21T21:20:03.39207Z","shell.execute_reply.started":"2021-09-21T21:20:03.384405Z","shell.execute_reply":"2021-09-21T21:20:03.39133Z"}}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split


# %% [code] {"execution":{"iopub.status.busy":"2021-09-21T21:20:03.393788Z","iopub.execute_input":"2021-09-21T21:20:03.394416Z","iopub.status.idle":"2021-09-21T21:20:03.407097Z","shell.execute_reply.started":"2021-09-21T21:20:03.394362Z","shell.execute_reply":"2021-09-21T21:20:03.406337Z"}}
input_path = Path('datasets')
im_size = 320

# %% [code] {"execution":{"iopub.status.busy":"2021-09-21T21:20:03.408679Z","iopub.execute_input":"2021-09-21T21:20:03.40924Z","iopub.status.idle":"2021-09-21T21:20:03.769862Z","shell.execute_reply.started":"2021-09-21T21:20:03.409176Z","shell.execute_reply":"2021-09-21T21:20:03.768648Z"}}
image_files = list(input_path.glob('*.jpg'))

# %% [code] {"execution":{"iopub.status.busy":"2021-09-21T21:20:03.771919Z","iopub.execute_input":"2021-09-21T21:20:03.772336Z","iopub.status.idle":"2021-09-21T21:20:03.780207Z","shell.execute_reply.started":"2021-09-21T21:20:03.772262Z","shell.execute_reply":"2021-09-21T21:20:03.77919Z"}}
def read_file(fname):
    # Read image
    im = Image.open(fname)

    # Resize
    im.thumbnail((im_size, im_size))

    # Convert to numpy array
    im_array = np.asarray(im)

    # Get target
    target = int(fname.stem.split('_')[0])

    return im_array, target

# %% [code] {"execution":{"iopub.status.busy":"2021-09-21T21:20:03.783653Z","iopub.execute_input":"2021-09-21T21:20:03.784068Z","iopub.status.idle":"2021-09-21T21:20:28.411459Z","shell.execute_reply.started":"2021-09-21T21:20:03.784009Z","shell.execute_reply":"2021-09-21T21:20:28.410511Z"}}
images = []
targets = []

for image_file in image_files:
    image, target = read_file(image_file)
    
    images.append(image)
    targets.append(target)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-21T21:20:28.413874Z","iopub.execute_input":"2021-09-21T21:20:28.414167Z","iopub.status.idle":"2021-09-21T21:20:32.339362Z","shell.execute_reply.started":"2021-09-21T21:20:28.414115Z","shell.execute_reply":"2021-09-21T21:20:32.338395Z"}}
X = (np.array(images).astype(np.float32) / 127.5) - 1
y_cls = np.array(targets)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-21T21:20:32.340831Z","iopub.execute_input":"2021-09-21T21:20:32.341088Z","iopub.status.idle":"2021-09-21T21:20:32.350054Z","shell.execute_reply.started":"2021-09-21T21:20:32.341045Z","shell.execute_reply":"2021-09-21T21:20:32.348622Z"}}
X.shape, y_cls.shape

# %% [code] {"execution":{"iopub.status.busy":"2021-09-21T21:20:32.351622Z","iopub.execute_input":"2021-09-21T21:20:32.352153Z","iopub.status.idle":"2021-09-21T21:20:32.616442Z","shell.execute_reply.started":"2021-09-21T21:20:32.352098Z","shell.execute_reply":"2021-09-21T21:20:32.615689Z"}}
i = 555
plt.imshow(np.uint8((X[i] + 1) * 127.5))
plt.title(str(y_cls[i]));

# %% [code] {"execution":{"iopub.status.busy":"2021-09-21T21:20:32.617617Z","iopub.execute_input":"2021-09-21T21:20:32.618014Z","iopub.status.idle":"2021-09-21T21:20:32.626912Z","shell.execute_reply.started":"2021-09-21T21:20:32.617974Z","shell.execute_reply":"2021-09-21T21:20:32.625618Z"}}
coins_ids = {
    5: 0,
    10: 1,
    25: 2,
    50: 3,
    100: 4
}

ids_coins = [5, 10, 25, 50, 100]

y = np.array([coins_ids[coin] for coin in y_cls])

# %% [code] {"execution":{"iopub.status.busy":"2021-09-21T21:20:32.628418Z","iopub.execute_input":"2021-09-21T21:20:32.629113Z","iopub.status.idle":"2021-09-21T21:20:35.031449Z","shell.execute_reply.started":"2021-09-21T21:20:32.629064Z","shell.execute_reply":"2021-09-21T21:20:35.030741Z"}}
X_train, X_valid, y_train, y_valid, fname_train, fname_valid = train_test_split(
    X, y, image_files, test_size=0.2, random_state=42)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-21T21:20:35.032583Z","iopub.execute_input":"2021-09-21T21:20:35.032967Z","iopub.status.idle":"2021-09-21T21:20:35.040386Z","shell.execute_reply.started":"2021-09-21T21:20:35.032926Z","shell.execute_reply":"2021-09-21T21:20:35.039368Z"}}
im_width = X.shape[2]
im_height = X.shape[1]

im_width, im_height

# %% [markdown]
# # Keras

# %% [code] {"execution":{"iopub.status.busy":"2021-09-21T21:20:35.041946Z","iopub.execute_input":"2021-09-21T21:20:35.042205Z","iopub.status.idle":"2021-09-21T21:20:36.967384Z","shell.execute_reply.started":"2021-09-21T21:20:35.042158Z","shell.execute_reply":"2021-09-21T21:20:36.96612Z"}}
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, GlobalAvgPool2D, GlobalMaxPool2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

# %% [code] {"execution":{"iopub.status.busy":"2021-09-21T21:20:36.969111Z","iopub.execute_input":"2021-09-21T21:20:36.969516Z","iopub.status.idle":"2021-09-21T21:20:37.164035Z","shell.execute_reply.started":"2021-09-21T21:20:36.969443Z","shell.execute_reply":"2021-09-21T21:20:37.163028Z"}}
model = Sequential()

# CNN network
model.add( Conv2D(16, 3, activation='relu', padding='same', input_shape=(im_height, im_width, 3)) )
model.add( MaxPool2D(2) )

model.add( Conv2D(32, 3, activation='relu', padding='same') )
model.add( MaxPool2D(2) )

model.add( Conv2D(64, 3, activation='relu', padding='same') )
model.add( MaxPool2D(2) )

model.add( Conv2D(128, 3, activation='relu', padding='same') )
model.add( MaxPool2D(2) )

model.add( Conv2D(256, 3, activation='relu', padding='same') )

# Transition between CNN and MLP
model.add( GlobalAvgPool2D() )

# MLP network
model.add( Dense(256, activation='relu') )

model.add( Dense(5, activation='softmax') )

model.summary()

# %% [code] {"execution":{"iopub.status.busy":"2021-09-21T21:20:37.165569Z","iopub.execute_input":"2021-09-21T21:20:37.165897Z","iopub.status.idle":"2021-09-21T21:20:37.241852Z","shell.execute_reply.started":"2021-09-21T21:20:37.165848Z","shell.execute_reply":"2021-09-21T21:20:37.24071Z"}}
optim = Adam(lr=1e-3)
model.compile(optim, 'sparse_categorical_crossentropy', metrics=['acc'])

# %% [code] {"execution":{"iopub.status.busy":"2021-09-21T21:27:27.624513Z","iopub.execute_input":"2021-09-21T21:27:27.624855Z","iopub.status.idle":"2021-09-21T22:22:49.123122Z","shell.execute_reply.started":"2021-09-21T21:27:27.624814Z","shell.execute_reply":"2021-09-21T22:22:49.121723Z"}}
callbacks = [
    ReduceLROnPlateau(patience=5, factor=0.1, verbose=True),
    ModelCheckpoint('best.model', save_best_only=True),
    EarlyStopping(patience=12)
]

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid), batch_size=32,
                   callbacks=callbacks)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-21T21:26:52.657617Z","iopub.status.idle":"2021-09-21T21:26:52.65822Z"}}
df_history = pd.DataFrame(history.history)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-21T21:26:52.659871Z","iopub.status.idle":"2021-09-21T21:26:52.660461Z"}}
ax = df_history[['acc', 'val_acc']].plot()
ax.set_ylim(0.9, 1)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-21T21:26:52.662134Z","iopub.status.idle":"2021-09-21T21:26:52.662799Z"}}
df_history['val_acc'].max()

# %% [code] {"execution":{"iopub.status.busy":"2021-09-21T21:26:52.664656Z","iopub.status.idle":"2021-09-21T21:26:52.665241Z"}}
model.load_weights('best.model')

# %% [code] {"execution":{"iopub.status.busy":"2021-09-21T21:26:52.666593Z","iopub.status.idle":"2021-09-21T21:26:52.667255Z"}}
model.evaluate(X_valid, y_valid)

# %% [markdown]
# # Evaluate results

# %% [code] {"execution":{"iopub.status.busy":"2021-09-21T21:26:52.668532Z","iopub.status.idle":"2021-09-21T21:26:52.669127Z"}}
y_pred = model.predict(X_valid)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-21T21:26:52.670566Z","iopub.status.idle":"2021-09-21T21:26:52.671132Z"}}
y_pred_cls = y_pred.argmax(1)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-21T21:26:52.672534Z","iopub.status.idle":"2021-09-21T21:26:52.673084Z"}}
errors = np.where(y_pred_cls != y_valid)[0]
errors

# %% [code] {"execution":{"iopub.status.busy":"2021-09-21T21:26:52.674339Z","iopub.status.idle":"2021-09-21T21:26:52.674944Z"}}
i = 55
plt.figure(figsize=(10, 10))
im = Image.open(fname_valid[i])
plt.imshow(np.uint8(im), interpolation='bilinear')
plt.title('Class: {}, Predicted: {}'.format(ids_coins[y_valid[i]], ids_coins[np.argmax(y_pred[i])]));

# %% [code]
