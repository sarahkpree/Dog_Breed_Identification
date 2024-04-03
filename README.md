# Dog Breed Identification

## Table of Contents

- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Tools](#tools)
- [Handcrafted CNN Models](#handcrafted-cnn-models)
- [Transfer Learning Models](#transfer-learning-models)
- [Results](#results)
- [Limitations](#limitations)

### Project Overview

The aim of this group project was to develop models capable of categorizing various dog breeds using Convolutional Neural Networks (CNNs). With a dataset comprising of 120 dog breeds, we trained multiple models employing both handcrafted CNN architectures and transfer learning approaches. Keeping hyperparameters constant across all models during training, we found that transfer learning models consistently outperformed handcrafted CNN models on this dataset, with Xception achieving the highest accuracy rate of 80.5%. Additionally, we highlighted the significance of fine-tuning models, which greatly improves their performance. Furthermore, we examined and compared the GPU requirements and training times of different models, noting that transfer learning models, particularly ResNet50V2, required over 15 hours. Lastly, we also explored dog breeds that are easily distinguishable and those that are not.

### Data Sources

The Stanford Dogs dataset is comprised of approx 20,000 images of 120 dog breeds worldwide, sourced from ImageNet. With roughly 150 images per class, the dataset is balanced. The dataset is too large to be uploaded to the repository, but it can be found [here](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset).

### Tools

- Python: pandas, numpy, sklearn, seaborn, keras

### Handcrafted CNN Models

We created several Convolutional Neural Network (CNN) models to compare their performance. 

- **Model 1:** Basic CNN Model
- **Model 2:** Basic CNN Model with Batch Normalization
- **Model 3:** Basic CNN Model with Batch Normalization & L1 Regularization
- **Model 4:** Basic CNN Model with Batch Normalization & L2 Regularization
- **Model 5:** Basic CNN Model with Batch Normalization, L1 & L2 Regularization

The hyperparameters (optimizer, learning rate, epochs, batch size) were the same across all models. 

**Basic Architecture of all five handcrafted models:** 

```python
#model 1 -- dropout + max pooling only 
model_1=Sequential()
model_1.add(Conv2D(16, kernel_size=(5, 5), input_shape=(200, 200, 3)))
model_1.add(Activation('relu'))
model_1.add(MaxPooling2D(pool_size=(2,2)))
model_1.add(Dropout(0.5))
               
model_1.add(Conv2D(64, kernel_size=(3, 3)))
model_1.add(Activation('relu'))
model_1.add(MaxPooling2D(pool_size=(2,2)))
model_1.add(Dropout(0.5))

model_1.add(Conv2D(128, kernel_size=(3, 3)))
model_1.add(Activation('relu'))
model_1.add(MaxPooling2D(pool_size=(2,2)))
model_1.add(Dropout(0.5))
model_1.add(Flatten())

model_1.add(Dense(200, activation='relu'))
model_1.add(Dense(175, activation='relu'))
model_1.add(Dense(150, activation='relu'))
model_1.add(Dense(numClasses, activation='softmax'))
```
**Training the model:**

```python
#compile and train
MyEpochs = 50
opt = keras.optimizers.Adam(learning_rate=0.01)

model_1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt, 
              metrics=['accuracy']) 

model_1.fit(training_data,
                        batch_size = 32,
                        epochs = MyEpochs,
                        validation_data=test_data,
                        shuffle = 1)
```

### Transfer Learning Models

We created three different transfer learning models. By comparing the performance of these three transfer learning models, we gained insights into which models were best suited for our specific task and dataset.

- **Model 1:** VGG 16
- **Model 2:** Xception
- **Model 3:** ResNet5OV2

All preprocessing (image augmentation) and hyperparameters (optimizer, learning rate, epochs, batch size) were the same across all models. 

**Example architecture for Xception model:**

```python
# model architecture

cnn = tf.keras.applications.Xception(include_top=False, weights="imagenet", input_shape=(299, 299, 3))
cnn.trainable = False

flatten_layer = tf.keras.layers.Flatten()
dense_layer_1 = Dense(1024, activation='relu')
dense_layer_2 = Dense(786, activation='relu')
dense_layer_3 = Dense(345, activation='relu')
prediction_layer = Dense(120, activation='softmax')

Classifier = Sequential([
    cnn,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    dense_layer_3,
    prediction_layer
])
```
**Training the model:** 

```python
MyEpochs = 50
opt = keras.optimizers.Adam()

Classifier.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt, 
              metrics=['accuracy']) 

Classifier.fit(training_data,
                        batch_size = 32,
                        epochs = MyEpochs,
                        validation_data=test_data,
                        shuffle = 1)
```

### Results

The comparison of these eight models was based on accuracy, macro, and weighted average F1 scores, along with GPU training time. Xception demonstrated the highest accuracy at 80.5%, followed by ResNet50V2 at 55.6% and VGG16 at 36.7%. Transfer learning models notably outperformed handcrafted CNN models, which achieved only approximately 1% accuracy. However, despite the superior performance, transfer learning models required significantly longer training times compared to handcrafted CNN models, with ResNet50V2 taking over 15 hours. Additionally, Xception attained the best Macro average F1 score of 0.8 among all models, further highlighting the superiority of transfer learning models over handcrafted CNN models, which scored close to zero in F1 score.

<img width="851" alt="Screen Shot 2024-04-03 at 3 30 43 PM" src="https://github.com/sarahkpree/Dog_Breed_Identification/assets/61251211/ffd507ae-f468-43d3-b963-bc5ccffea3ed">



<img width="820" alt="Screen Shot 2024-04-03 at 3 30 24 PM" src="https://github.com/sarahkpree/Dog_Breed_Identification/assets/61251211/11fa9fd9-9d8c-411d-b8a0-ff52ca41d322">

### Limitations

Some potential limitations of this project include:
- Some images contain distractions like humans and objects, while certain dog characteristics, such as similarities between breeds and variations in age, posed challenges for training.
- The choice of models evaluated in the project may not encompass all potential architectures or techniques that could improve performance.
- The models' hyperparameters may not have been optimized thoroughly, potentially leaving room for improvement in their performance.
- The reported training times were based on GPU performance, which may vary depending on the specific hardware used. This could limit the reproducibility of results on different computing platforms or environments.
