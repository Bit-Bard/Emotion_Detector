import tkinter as tk
from tkinter import filedialog, Label, Button
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import numpy as np
import cv2

top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)
facec = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

try:
    model = load_model("model_full.keras")  # Use "model_full.h5" if saved in HDF5 format
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def Detect(file_path):
    try:
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError("Unable to read the image file.")

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_image, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_image[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis, :, :, np.newaxis]))]
            print("Predicted Emotion is " + pred)
            label1.configure(foreground="#011638", text=pred)
    except Exception as e:
        print(f"Error detecting emotion: {e}")
        label1.configure(foreground="#011638", text="Unable to detect")

def show_Detect_button(file_path):
    detect_b = Button(top, text="Detect Emotion", command=lambda: Detect(file_path), padx=10, pady=5)
    detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_Detect_button(file_path)
    except Exception as e:
        print(f"Error uploading image: {e}")


upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5) # Upload button
upload.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='bottom', pady=50)

sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')

heading = Label(top, text='Emotion Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()
top.mainloop()