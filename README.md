# Gesture-Synthesizer
This project aims to create a simple gesture synthesizer that generates gestures based on input text.

We design our workflow into two parts.
1. Input text process: We utilize large-language-pretrained-model to process the input text, understanding the inner emotions. Specifically, the system will classify the user input into several emotion categories for further processing.
2. Gesture Generation: We used Maya to create a gesture model. And the system will generate the appropriate gesture video based on the input text and display it on the screen.

Package requirements are lists in requirements.txt
You can train the model by running the bert_train.py or skip training and generate gestures by running the gesture-synthesizer.py

**Tips: No matter which code you want to run, you need to download the pre-trained model and place it to path `./models/bert_`**

## Get the models
Download the pre-trained model and place it to path `./models/bert_`
https://drive.google.com/drive/folders/1J_yLCYZ9gjUnQrjwEHK_hv6_M59E-Pfj?usp=sharing

## Train the model
Go the the project root directory and run the following command
```python3 ./bert_train.py```

The model will be saved to `./models/bert_`

## Test the model
Go the the project root directory and run the following command
```python3 ./bert_test.py```

You can change the input text in the code to test the model.

## Generate gestures
Go the the project root directory and run the following command
```python3 ./gesture-synthesizer.py```

You can change the input text in the input-box to generate different gestures.

**Tips: Considering different performance of your computer, the model may load from 10 seconds to minutes. If there is no responding when you click the buttom, please wait one minute for model loading.**