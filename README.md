# yolo-img-classifier_service

<p align="center">
<img src="https://github.com/leepand/yolo-img-classifier/blob/master/static/img/screenshot.png" alt="Drawing" style="width:40%;"/>
</p>

[**Demo**](http://knowledge.wanda.cn:5003)

A Flask  Web Interface for [yolo](https://github.com/pjreddie/darknet) Image Classifier.

This app simply invoked the [pre-trained model](http://pjreddie.com/media/files/yolo.weights) provided by darknet. 

## Deployment

### Step - 1: Environment
```bash

sudo pip install Flask
sudo pip install keras
sudo yum install numpy opencv*


sudo pip install gunicorn
```

### Step - 2: Clone This Project

```bash
git clone https://github.com/leepand/yolo-img-classifier.git
```

### Step - 3: Download Pre-Trained yolo Model

From http://pjreddie.com/media/files/, download

- wget http://pjreddie.com/media/files/yolo.weights


Note that we need to put all these three files under application directory.

### Step - 4: Start Service

```bash
sudo gunicorn -b 0.0.0.0:80 app:app
```
