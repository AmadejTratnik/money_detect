# Monetary Gift Detector
+ Prompt generated dataset of ~**1000 samples** that classifies monetary vs. non-monetary gifts.
+ 6 models trained via cross validation over whole dataset, **SVM with TFIDF vectorizer** performed the best.

## Installation
Download the project from Github and change your current directory:
```
$ (base) cd money_detect
```
Use a virtual environment to isolate your environment, and install the required dependencies.
```
$ (base) python3 -m venv venv
$ (base) source venv/bin/activate
$ (venv) pip3 install -r requirements.txt
```

## (New) Model Training
To add a new model (add model with its vectorizer to **models.py**) or just train again with a new/updated dataset, write
```
$ (venv) python3 pipeline.py
```
The script automatically loops over all models, fits them to the dataset, and calculates confusion matrices, ROC/AUC curves and CV scores. 

Model with the best CV score is saved with ONNX format as **final_model.onnx**.

## Inference
When *final_model.onnx* is present, write
```
$ (venv) python3 load_use_model.py
```
And the inference will start for its input text.

## Training results
![Confusion matrices](https://github.com/AmadejTratnik/money_detect/blob/main/images/confusion_matrices.png)
![ROC AUC curves](https://github.com/AmadejTratnik/money_detect/blob/main/images/roc_curves.png)
