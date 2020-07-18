#import tensorflow as tf
import keras
import gradio as gr
from keras.models import Sequential

(x_train ,  y_train) , (x_test , y_test) = keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0


model = Sequential([
	    keras.layers.Flatten(input_shape = (28 , 28)) ,
	    keras.layers.Dense(128 ,  activation = 'relu') ,
	    keras.layers.Dense(10 , activation  = 'softmax' )])

model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy' ,  metrics = ['accuracy'])

model.fit(x_train , y_train , validation_data = (x_test , y_test) , epochs = 10) 

def Classify(image):
	prediction = model.predict(image).tolist()[0]
	return {str(i): prediction[i] for i in range(10)}

sketchpad = gr.inputs.Sketchpad()
label = gr.outputs.Label(num_top_classes = 3)
interface  = gr.Interface(Classify , sketchpad , label , live= True , capture_session = True)

interface.launch()
