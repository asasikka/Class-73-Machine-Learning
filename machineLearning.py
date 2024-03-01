import cv2
import numpy as np

from keras.models import load_model
model=load_model("keras_model.h5", compile=False)

classNames = open("labels.txt", "r").readlines()
print(classNames)
print(model)

camera=cv2.VideoCapture(0)

while True:
    ret,image=camera.read()
    image_fliped=cv2.flip(image,1)

    image=cv2.resize(image_fliped,(224,224), interpolation=cv2.INTER_AREA)
    
    try:
        image=cv2.putText(image,classNames[2:], (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        image=cv2.putText(image,str(np.round(confidenceScore*100)[:-2],(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255,),2))

    except:
        pass
    
    cv2.imshow("Machine Learning Window", image)

    image= np.asarray(image,dtype=np.float32).reshape(1,224,224,3)
    
    image=(image/127.5)-1
    # print(image)
    


    prediction = model.predict(image)
    index=np.argmax(prediction)

    className= classNames[index]
    print(className)

    confidenceScore = prediction[0][index]
    print("Conf", confidenceScore)

    if cv2.waitKey(25)==32:
        break

camera.release()
cv2.destroyAllWindows()