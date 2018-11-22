from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import sys
import cv2
i=1
cap = cv2.VideoCapture(0)
model=load_model(sys.argv[1])
while(True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.imwrite(sys.argv[6] + str(i) + '.jpeg', frame)
    testgen = ImageDataGenerator(rescale=1. / 255)
    now_generator = testgen.flow_from_directory(sys.argv[6], target_size=(150, 150), batch_size=1,
                                                class_mode='binary')
    a=1-(model.predict_generator(now_generator, 1//1))
    print(a)
    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break
cap.release()
cv2.destroyAllWindows()
