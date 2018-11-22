from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout,Flatten,Dense
from keras.models import load_model
import sys
import cv2
i=1
cap = cv2.VideoCapture(0)
train_dir=sys.argv[3]
val_dir=sys.argv[4]
test_dir=sys.argv[5]
img_width, img_height=150,150
input_share=(img_width,img_height,3)
epstring=int(sys.argv[2])
epochs=epstring
batch_size=10
nb_train_samples=1300
nb_validation_samples=280
nb_test_samples=280

model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=input_share))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

datagen=ImageDataGenerator(rescale=1./255)
train_generator=datagen.flow_from_directory(train_dir,
                                            target_size=(img_width, img_height),
                                            batch_size=batch_size, class_mode='binary')
val_generator=datagen.flow_from_directory(val_dir,
                                          target_size=(img_width,img_height),
                                          batch_size=batch_size, class_mode='binary')
test_generator=datagen.flow_from_directory(test_dir,
                                           target_size=(img_width, img_height),
                                           batch_size=batch_size, class_mode='binary')
testgen=ImageDataGenerator(rescale=1./255)

model.fit_generator(train_generator,steps_per_epoch=nb_train_samples//batch_size,epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=nb_validation_samples//batch_size)
scores=model.evaluate_generator(test_generator, nb_test_samples//batch_size)
print("Точность на тестовых данных: %.2f%%"%(scores[1]*100))
model.save(sys.argv[1])
