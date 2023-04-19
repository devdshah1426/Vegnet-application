import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.xception import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
  

# Define a dictionary of vegetables and their corresponding file paths
vegetables = {'Bell_Pepper': r"D:\GItHub\Classname\VCQ_Classname\Bell_Pepper",
              'Chile_Pepper': r"D:\GItHub\Classname\VCQ_Classname\Chile_Pepper",
              'Tomato': r"D:\GItHub\Classname\VCQ_Classname\Tomato",
              'New_Mexico_Green_Chile':r"D:\GItHub\Classname\VCQ_Classname\New_Mexico_Green_Chile"}

# Create the GUI window
root = tk.Tk()
root.title('Accuracy of VQC_Xception Model')

"""# Create a dropdown menu to select the vegetable
selected_vegetable = tk.StringVar()
vegetable_label = tk.Label(root, text='Select a vegetable:')
vegetable_label.pack()
vegetable_dropdown = tk.OptionMenu(root, selected_vegetable, *vegetables.keys())
vegetable_dropdown.pack()
# Function to load and display the selected image"""
def load_image():
    
    # Load the corresponding model for the selected vegetable
    model_path = r'D:\GItHub\Trained Model\Vegetable.h5'
    model = tf.keras.models.load_model(model_path)
    
    # Load the corresponding training data generator
    train_data_gen_1 = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_generator_1= train_data_gen_1.flow_from_directory(
    r"D:\GItHub\Classname\VCQ_Classname",
    target_size=(299, 299),
    batch_size=10,
    class_mode='categorical',
    shuffle=True)
    
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

    # Preprocess the image
    
    img_array = img_to_array(image)
    img_array = preprocess_input(img_array)
    
    # Make a prediction using the model
    pred = model.predict(np.array([img_array]))
     
    # Get the predicted class label
    class_labels_1 = train_generator_1.class_indices
    class_label_1 = list(class_labels_1.keys())[list(class_labels_1.values()).index(np.argmax(pred))]
    
    # Display the predicted class label in the GUI
    #prediction_label.configure(text='Predicted class: ' + class_label_1)
        
    # Load the corresponding model for the selected vegetable
    model_path = f'D:\\GItHub\\Trained Model\\{class_label_1}.h5'
    model = tf.keras.models.load_model(model_path)
    
    # Load the corresponding training data generator
    train_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_generator = train_data_gen.flow_from_directory(
        vegetables[class_label_1],
        target_size=(299, 299),
        batch_size=10,
        class_mode='categorical',
        shuffle=True)
    # Make a prediction using the model
    pred = model.predict(np.array([img_array]))
    
    # Get the predicted class label
    class_labels = train_generator.class_indices
    class_label = list(class_labels.keys())[list(class_labels.values()).index(np.argmax(pred))]
    
    # Display the predicted class label in the GUI
    prediction_label.configure(text='Predicted class: ' + class_label)
    

# Create the image label
image_label = tk.Label(root)
image_label.pack()

# Create the button to load the image
load_button = tk.Button(root, text='Load Image', command=load_image)
load_button.pack()

# Create the label to display the predicted class label
prediction_label = tk.Label(root)
prediction_label.pack()

# Run the GUI
root.mainloop()
