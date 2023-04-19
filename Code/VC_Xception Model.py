import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.xception import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

train_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_data_gen.flow_from_directory(
    r"D:\GItHub\Classname\VC_Classname",
    target_size=(299, 299),
    batch_size=10,
    class_mode='categorical',
    shuffle=True
)
# Create the GUI window
root = tk.Tk()
root.title('Accuracy of VC_Xception Model')
root.configure(bg='yellow')

# Create the image label
image_label = tk.Label(root, bg='yellow', fg='black')
image_label.pack()

# Create the label to display the predicted class label
prediction_label = tk.Label(root, bg='yellow', fg='black')
prediction_label.pack()

# Load the saved model
model = tf.keras.models.load_model(r'D:\GItHub\Trained Model\VC_Xception.h5')

# Define a function to load and display the selected image and make a prediction
def load_image():
    # Get the file path of the selected image
    file_path = filedialog.askopenfilename()
    
    # Load the image using PIL
    image = Image.open(file_path)
    
    # Resize the image to (299, 299)
    image = image.resize((299, 299))
    
    # Display the image in the GUI
    photo = ImageTk.PhotoImage(image)
    image_label.configure(image=photo)
    image_label.image = photo

    # Convert the image to a numpy array
    img_array = img_to_array(image)

    # Preprocess the image
    img_array = preprocess_input(img_array)
    
    # Make a prediction using the model
    pred = model.predict(np.array([img_array]))
    
    # Get the predicted class label and number
    #print(train_generator.class_indices)
    class_labels = train_generator.class_indices
    class_index = np.argmax(pred)
    class_label = list(class_labels.keys())[class_index]
    class_num = class_index + 1
    
    
    
    if class_num==1:
        prediction_label.configure(text='Damaged Bell Pepper' )
    elif class_num==2:
        prediction_label.configure(text='Dried Bell Pepper' )
    elif class_num==3:
        prediction_label.configure(text='Old Bell Pepper' )
    elif class_num==4:
        prediction_label.configure(text='Ripe Bell Pepper' )
    elif class_num==5:
        prediction_label.configure(text='Unriped Bell Pepper' )
    elif class_num==6:
        prediction_label.configure(text='Damaged Chile Pepper' )
    elif class_num==7:
        prediction_label.configure(text='Dried Chile Pepper' )
    elif class_num==8:
        prediction_label.configure(text='Old Chile Pepper' )
    elif class_num==9:
        prediction_label.configure(text='Ripe Chile Pepper' )
    elif class_num==10:
        prediction_label.configure(text='Unriped Chile Pepper' )
    elif class_num==11:
        prediction_label.configure(text='Damaged New Mexico Green Chile' )
    elif class_num==12:
        prediction_label.configure(text='Dried New Mexico Green Chile' )
    elif class_num==13:
        prediction_label.configure(text='Old New Mexico Green Chile' )
    elif class_num==14:
        prediction_label.configure(text='Riped New Mexico Green Chile' )
    elif class_num==15:
        prediction_label.configure(text='Unriped New Mexico Green Chile' )
    elif class_num==16:
        prediction_label.configure(text='Damaged Tomato' )
    elif class_num==17:
        prediction_label.configure(text='Old Tomato' )
    elif class_num==18:
        prediction_label.configure(text='Riped Tomato' )
    elif class_num==19:
        prediction_label.configure(text='Unripe Tomato' )
    

# Create the button to load the image and make a prediction
load_button = tk.Button(root, text='Load Image', command=load_image)
load_button.pack()

# Run the GUI
root.mainloop()
