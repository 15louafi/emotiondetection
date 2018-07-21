import pandas as pd #Dataframes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns #For graphs and neat visualization

from scipy.io import loadmat

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization #The layers we will be using
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator #For data augmentation
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils import np_utils
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from utils.datasets import DataManager
from utils.datasets import split_imdb_data

def train_emotion():
    data = pd.read_csv('./data/fer2013.csv')
    X_train = data['pixels']
    X_train = [ dat.split() for dat in X_train]
    X_train = np.array(X_train)
    X_train = X_train.astype('float64')
    
    Y_train = data["emotion"]
    
    X_train = X_train / 255.0
    
    
    X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
    
    Y_train = np_utils.to_categorical(Y_train)
    # 
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
    
    
    model = Sequential()
    
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu', input_shape = (48,48,1)))
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters = 86, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu'))
    model.add(Conv2D(filters = 86, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    #model.add(Dense(1024, activation = "relu"))
    #model.add(Dropout(0.5))
    model.add(Dense(512, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation = "softmax"))
    
    optimizer = RMSprop(lr=0.001,
                        rho=0.9,
                        epsilon=1e-08, 
                        decay=0.0)
    model.compile(optimizer = optimizer ,
                loss = "categorical_crossentropy",
                metrics=["accuracy"]) 
    
    
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                                patience=3, 
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=0.00001)
    
    
    epochs = 30 
    batch_size = 86
    
    
    datagen = ImageDataGenerator(
            featurewise_center=False,  
            samplewise_center=False,  
            featurewise_std_normalization=False,  
            samplewise_std_normalization=False,  
            zca_whitening=False,  
            rotation_range=15,  
            zoom_range = 0.1,  
            width_shift_range=0.1, 
            height_shift_range=0.1,  
            horizontal_flip=False,  
            vertical_flip=False)  
    
    datagen.fit(X_train)
    
    
    history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                                epochs = epochs, validation_data = (X_val,Y_val),
                                verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                                , callbacks=[learning_rate_reduction])
    
    
    #save the model weights
    import h5py
    json_string = model.to_json()
    model.save_weights('./models/Face_model_weights.h5')
    open('./models/Face_model_architecture.json', 'w').write(json_string)
    model.save_weights('./models/Face_model_weights.h5')
    score = model.evaluate(X_train, Y_train, verbose=0)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])

def load_data(mat_path):
    d = loadmat(mat_path)
    return d["image"], d["gender"][0], d["age"][0], d["db"][0], d["img_size"][0, 0], d["min_score"][0, 0]
    
def train_gender():
    # parameters
    batch_size = 32
    num_epochs = 1000
    validation_split = .2
    do_random_crop = False
    num_classes = 2
    dataset_name = 'imdb'
    input_shape = (64, 64, 1)
    if input_shape[2] == 1:
        grayscale = True
    images_path = '../datasets/imdb_crop/'
    log_file_path = '../models/gender_training.log'
    trained_models_path = '../models/gender'
    
    # data loader
    image, gender, _, _, image_size, _ = load_data(images_path)
    X_data = image
    y_data_g = np_utils.to_categorical(gender, 2)
    
    data_num = len(X_data)
    indexes = np.arange(data_num)
    np.random.shuffle(indexes)
    X_data = X_data[indexes]
    Y_data = y_data_g[indexes]
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = validation_split, random_state=2)

    datagen = ImageDataGenerator(
            featurewise_center=False,  
            samplewise_center=False,  
            featurewise_std_normalization=False,  
            samplewise_std_normalization=False,  
            zca_whitening=False,  
            rotation_range=15,  
            zoom_range = 0.1,  
            width_shift_range=0.1, 
            height_shift_range=0.1,  
            horizontal_flip=False,  
            vertical_flip=False) 
    
    # model parameters/compilation
    model = Sequential()
    
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu', input_shape = input_shape))
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters = 86, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu'))
    model.add(Conv2D(filters = 86, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    #model.add(Dense(1024, activation = "relu"))
    #model.add(Dropout(0.5))
    model.add(Dense(512, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation = "softmax"))
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    model.summary()
    
    # model callbacks
    early_stop = EarlyStopping('val_loss', patience=patience)
    ReduceLROnPlateau(monitor='val_acc', 
                                                patience=50, 
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=0.00001)
    
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names,
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=False)
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]
    datagen.fit(X_train)
    # training model
    classifier = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                                epochs = epochs,
                                   steps_per_epoch=train_num // batch_size,
                                   validation_data=(X_val, Y_val),
                                   epochs=nb_epochs, verbose=1,
callbacks=callbacks)
    
def train_age():
    # parameters
    batch_size = 32
    num_epochs = 1000
    validation_split = .1
    input_shape = (64, 64, 1)
    if input_shape[2] == 1:
        grayscale = True
    images_path = '../datasets/imdb_crop/'
    log_file_path = '../models/age_training.log'
    trained_models_path = '../models/age/'
    
    # data loader
    image, _, age, _, image_size, _ = load_data(images_path)
    X_data = image
    y_data_a = np_utils.to_categorical(age, 101)
    
    data_num = len(X_data)
    indexes = np.arange(data_num)
    np.random.shuffle(indexes)
    X_data = X_data[indexes]
    Y_data = y_data_a[indexes]
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = validation_split, random_state=2)

    datagen = ImageDataGenerator(
            featurewise_center=False,  
            samplewise_center=False,  
            featurewise_std_normalization=False,  
            samplewise_std_normalization=False,  
            zca_whitening=False,  
            rotation_range=15,  
            zoom_range = 0.1,  
            width_shift_range=0.1, 
            height_shift_range=0.1,  
            horizontal_flip=False,  
            vertical_flip=False) 
    
    # model parameters/compilation
    model = Sequential()
    
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu', input_shape = input_shape))
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters = 86, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu'))
    model.add(Conv2D(filters = 86, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    #model.add(Dense(1024, activation = "relu"))
    #model.add(Dropout(0.5))
    model.add(Dense(512, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(101, activation = "softmax"))
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    model.summary()
    
    # model callbacks
    early_stop = EarlyStopping('val_loss', patience=patience)
    ReduceLROnPlateau(monitor='val_acc', 
                                                patience=50, 
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=0.00001)
    
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names,
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=False)
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]
    datagen.fit(X_train)
    # training model
    classifier = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                                epochs = epochs,
                                   steps_per_epoch=train_num // batch_size,
                                   validation_data=(X_val, Y_val),
                                   epochs=nb_epochs, verbose=1,
callbacks=callbacks)


    # json_string = model.to_json()
    # model.save_weights('./models/Ageweights.h5')
    # open('./models/Age_arch.json', 'w').write(json_string)
    # model.save_weights('./models/Age_arch.h5')
    # score = model.evaluate(X_train, Y_train, verbose=0)
    # print('Test loss: ', score[0])
    # print('Test accuracy: ', score[1])

    # fig, ax = plt.subplots(2,1)
    # ax[0].plot(history.history['loss'], color='black', label="Training loss")
    # ax[0].plot(history.history['val_loss'], color='red', label="Validation loss",axes =ax[0])
    # legend = ax[0].legend(loc='best')
    # 
    # ax[1].plot(history.history['acc'], color='black', label="Training accuracy")
    # ax[1].plot(history.history['val_acc'], color='red',label="Validation accuracy")
    # legend = ax[1].legend(loc='best')
    # 
    # 
    # 
    # 

    # model.save('history.h5')
    # # 
    # # def plot_confusion_matrix(cm, classes,
    # #                           normalize=False,
    # #                           title='Confusion matrix',
    # #                           cmap=plt.cm.Blues):
    # #     plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # #     plt.title(title)
    # #     plt.colorbar()
    # #     tick_marks = np.arange(len(classes))
    # #     plt.xticks(tick_marks, classes, rotation=45)
    # #     plt.yticks(tick_marks, classes)
    # # 
    # #     if normalize:
    # #         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # # 
    # #     thresh = cm.max() / 2.
    # #     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    # #         plt.text(j, i, cm[i, j],
    # #                  horizontalalignment="center",
    # #                  color="white" if cm[i, j] > thresh else "black")
    # # 
    # #     plt.tight_layout()
    # #     plt.ylabel('True label')
    # #     plt.xlabel('Predicted label')
    # 
    # # Predict the values from the validation dataset
    # Y_pred = model.predict(X_val)
    # # Convert predictions classes to one hot vectors 
    # Y_pred_classes = np.argmax(Y_pred,axis = 1) 
    # # Convert validation observations to one hot vectors
    # Y_true = np.argmax(Y_val,axis = 1) 
    # # compute the confusion matrix
    # confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
    # # plot the confusion matrix
    # plot_confusion_matrix(confusion_mtx, classes = range(10)) 
    # 
    # 
    # 
    # #cell 35
    # errors = (Y_pred_classes - Y_true != 0)
    # 
    # Y_pred_classes_errors = Y_pred_classes[errors]
    # Y_pred_errors = Y_pred[errors]
    # Y_true_errors = Y_true[errors]
    # X_val_errors = X_val[errors]
    
    # def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    #     n = 0
    #     nrows = 2
    #     ncols = 3
    #     fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    #     for row in range(nrows):
    #         for col in range(ncols):
    #             error = errors_index[n]
    #             ax[row,col].imshow((img_errors[error]).reshape((28,28)))
    #             ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
    #             n += 1
    # 
    # Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)
    # true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))
    # delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors
    # sorted_dela_errors = np.argsort(delta_pred_true_errors)
    # most_important_errors = sorted_dela_errors[-6:]
    # display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
    
    
    # layer_names = []
    # for layer in model.layers[:-1]:
    #     layer_names.append(layer.name) 
    # images_nb_row = 10
    # for layer_name, layer_activation in zip(layer_names, activations):
    #     if layer_name.startswith('conv'):
    #         n_features = layer_activation.shape[-1]
    #         size = layer_activation.shape[1]
    #         n_cols = n_features // images_nb_row
    #         display_grid = np.zeros((size * n_cols, images_per_row * size))
    #         for col in range(n_cols):
    #             for row in range(images_per_row):
    #                 channel_image = layer_activation[0,:, :, col * images_nb_row + row]
    #                 channel_image -= channel_image.mean()
    #                 channel_image /= channel_image.std()
    #                 channel_image *= 64
    #                 channel_image += 128
    #                 channel_image = np.clip(channel_image, 0, 255).astype('uint8')
    #                 display_grid[col * size : (col + 1) * size,
    #                              row * size : (row + 1) * size] = channel_image
    #         scale = 1. / size
    #         plt.figure(figsize=(scale * display_grid.shape[1],
    #                             scale * display_grid.shape[0]))
    #         plt.title(layer_name)
    #         plt.grid(False)
    #         plt.imshow(display_grid, aspect='auto', cmap='viridis')
    # 
    # 
    # 
