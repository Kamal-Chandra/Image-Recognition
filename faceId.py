# Kivy Dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput

from kivy.clock import Clock
from kivy.logger import Logger
from kivy.graphics.texture import Texture

# Other Dependencies
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Distance Layer
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super(L1Dist, self).__init__(**kwargs)
    def call(self, inputs):
        input_embedding, validation_embedding = inputs
        return tf.math.abs(input_embedding - validation_embedding)

class CamApp(App):
    def build(self):
        self.web_cam = Image(size_hint=(1,.8), allow_stretch=True)
        self.register_button = Button(text="Register", on_press=self.register, size_hint=(1,.1), background_color=(0.9, 0.7, 0.2, 1))
        self.verify_button = Button(text="Verify", on_press=self.verify, size_hint=(1,.1), background_color=(0.9, 0.7, 0.2, 1))
        self.name_input = TextInput(hint_text="Enter your name", size_hint=(1,.1), font_size=20)
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1,.1), font_size=20, color=(0.2, 0.5, 0.8, 1))

        layout = BoxLayout(orientation='vertical', padding=20, spacing=10)
        layout.add_widget(self.web_cam)
        layout.add_widget(self.name_input)
        layout.add_widget(self.register_button)
        layout.add_widget(self.verify_button)
        layout.add_widget(self.verification_label)

        self.model = tf.keras.models.load_model('siameseModel.h5', custom_objects={'L1Dist': L1Dist})

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return layout

    def update(self, *args):
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    def preprocess(self, file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img, (100,100))
        img = img / 255.0
        return img

    def register(self, *args):
        name = self.name_input.text
        if not os.path.exists(os.path.join('application_data', 'verification_images', name)):
            os.makedirs(os.path.join('application_data', 'verification_images', name))
        for i in range(50):
            ret, frame = self.capture.read()
            frame = frame[120:120+250, 200:200+250, :]
            cv2.imwrite(os.path.join('application_data', 'verification_images', name, f'{i}.jpg'), frame)
            self.verification_label.text = 'Registering: Smile Please'
            Clock.usleep(600000)
        self.verification_label.text = f'Registered: {name}'

    def verify(self, *args):
        detection_threshold = 0.99
        verification_threshold = 0.8

        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)

        users = os.listdir(os.path.join('application_data', 'verification_images'))
        best_match = None
        best_match_score = 0

        for user in users:
            user_path = os.path.join('application_data', 'verification_images', user)
            images = os.listdir(user_path)
            scores = []
            for image in images:
                input_img = self.preprocess(SAVE_PATH)
                validation_img = self.preprocess(os.path.join(user_path, image))
                result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
                scores.append(result)
            avg_score = np.mean(scores)
            if avg_score > best_match_score:
                best_match_score = avg_score
                best_match = user

        if best_match_score > verification_threshold:
            self.verification_label.text = f'Verified: {best_match}'
        else:
            self.verification

if __name__ == '__main__':
    CamApp().run()
