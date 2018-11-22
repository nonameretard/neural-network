import sys
from keras.preprocessing.image import ImageDataGenerator
keras.models.load_model(sys.argv[1])
import cv2
i=1
now_dir=sys.argv[6]
img_width, img_height=150,150
cap = cv2.VideoCapture(0)
a=model.predict_generator(test_generator, nb_test_samples//batch_size)
while(True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.imwrite(sys.argv[6] + str(i) + '.jpeg', frame)
    testgen = ImageDataGenerator(rescale=1. / 255)
    now_generator = testgen.flow_from_directory(now_dir, target_size=(img_width, img_height), batch_size=1,
                                                class_mode='binary')
    a=1-(model.predict_generator(now_generator, 1//1))
    print(a)
    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break
cap.release()
cv2.destroyAllWindows()
