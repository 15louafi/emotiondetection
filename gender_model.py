from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from utils.datasets import DataManager
from models.cnn import mini_XCEPTION
from utils.data_augmentation import ImageGenerator
from utils.datasets import split_imdb_data

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
trained_models_path = '../models/'

# data loader
data_loader = DataManager(dataset_name)
ground_truth_data = data_loader.get_data()
train_keys, val_keys = split_imdb_data(ground_truth_data, validation_split)
image_generator = ImageGenerator(ground_truth_data, batch_size,
                                 input_shape[:2],
                                 train_keys, val_keys, None,
                                 path_prefix=images_path,
                                 vertical_flip_probability=0,
                                 grayscale=grayscale,
                                 do_random_crop=do_random_crop)

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

# training model
model.fit_generator(image_generator.flow(mode='train'),
                    steps_per_epoch=int(len(train_keys) / batch_size),
                    epochs=num_epochs, verbose=1,
                    callbacks=callbacks,
                    validation_data=image_generator.flow('val'),
validation_steps=int(len(val_keys) / batch_size))