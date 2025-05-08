# 🐱🐶 Cat vs. Dog Image Classifier

A simple image classification project built with TensorFlow and Keras that trains a model to distinguish between cats and dogs. This project is designed to predict whether a given image is a cat or a dog with a target accuracy of **at least 63%** on a 50-image test set.

> 🧠 **This challenge was provided by [freeCodeCamp’s Machine Learning with Python course](https://www.freecodecamp.org/learn/machine-learning-with-python/).**


## 🛠 What I Did

- Used TensorFlow/Keras to build a CNN from scratch
- Preprocessed image data using `ImageDataGenerator`
- Trained the model on labeled images of cats and dogs
- Tested it on 50 unseen images
- Visualized the results with the model's confidence (e.g., “72.35% dog”)

## 🤔 What I Learned

Throughout the development of this project, I encountered several practical challenges:

- **Directory structure issues**: Since `flow_from_directory` expects subdirectories for each class, I had to manually create an `unknown` subdirectory inside `test` and move all test images into it. 
- **Model architecture adjustments**: My first few training attempts hovered around ~50% accuracy. To improve this, I experimented with the transformations in `ImageDataGenerator` and also added an extra Conv2D/MaxPooling layer, which led to noticeable improvements.

## 🚀 Future Improvements

- **Transfer Learning**: Adapting a pre-trained model could improve the accuracy further by leveraging weights learned on large-scale image datasets.
- **Data Augmentation**: I could use more transformations, like changing the brightness or adding a vertical shift. The more varied the data, the better the model can generalize to new, unseen images and the less likely it is to overfit to the specific details of the training set.
- **Hyperparameter Tuning**: Adjusting different hyperparameters such as the learning rate, batch size, etc., could lead to better performance.





