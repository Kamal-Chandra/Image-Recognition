# Siamese Neural Network for Facial Verification

This project implements Siamese Neural Network for one-shot image recognition (detailed in `Image_Recognition.ipynb`) and a Kivy-based application for facial verification.

## Files Included

- **Image Recognition Model**: `Image_Recognition.ipynb`
- **Kivy App Implementation**: `faceid.py`, `layers.py`

## Getting Started

Follow these steps to set up and use the project:

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/Kamal-Chandra/Image-Recognition.git
cd Image-Recognition
```

### 2. Install Dependencies

Install the necessary dependencies, such as TensorFlow, Keras, OpenCV, Kivy, Matplotlib, Numpy.

```bash
pip install tensorflow keras opencv-python kivy matplotlib numpy
```

### 3. Train the Image Recognition Model

Open and execute all code blocks in the `Image_Recognition.ipynb` notebook to train the image recognition model. The trained model will be saved as `siameseModel.h5`.

### 4. Create Required Folders

Create a folder named `application_data` inside the repository folder. Inside `application_data`, create two subfolders named `input_image` and `verification_images`:

```bash
mkdir -p application_data/input_image application_data/verification_images
```

### 5. Run the Kivy Application

Run the Kivy application using:

```bash
python faceid.py
```

### 6. Using the App

- **Register New Users**: Use the "Register" button to register new users by capturing their images.
- **Verify Users**: Use the "Verify" button to verify if the user data is present by capturing and comparing the image.

## Acknowledgments

**Research Paper**

[Koch, G., Zemel, R., & Salakhutdinov, R. (2015). "Siamese Neural Networks for One-shot Image Recognition"](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

**Dataset**

[Labeled Faces in the Wild (LFW) Dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz)
