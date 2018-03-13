# -*- coding: UTF-8 -*-
"""Run a YOLO_v2 style detection model on test images."""
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from collections import namedtuple
import  hashlib
import datetime

import logging
try:
    import requests
except ImportError:
    # in rare cases requests may be not installed
    pass




import argparse
import colorsys
import imghdr
import os
import random
from utils.convert_result import convert_result, draw_helper
import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
#from darkflow.net.build import TFNet
import cv2
import time

import tensorflow as tf


_graph = None

def get_graph():
    global _graph

    if _graph == None:
        print('Graph is None. Init graph.')

        _graph = _sess.graph
    
    return _graph

_sess = None
_model_path = 'model_data/model.h5'


def _load_model(sess, model_dir):
    """Load facenet model
    """
    global _yolo_model
    # measure time used for loading model
    start = time.time()
    print('Loading models. Waiting...')

    
    g = get_graph()
    with g.as_default():
        _yolo_model = load_model(model_dir)
    #return _yolo_model

    print('Models loaded. time_used: ', time.time()-start)
def init_session():
    global _sess,_yolo_model

    # single session for facenet
    _sess = K.get_session()

    _load_model(_sess, _model_path)


def get_session():
    global _sess,_yolo_model

    if _sess == None:
        print('Session is None. Initialize session.')
        init_session()

    return _sess,_yolo_model


app = Flask(__name__)
# restrict the size of the file uploaded
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024


################################################
# Error Handling
################################################

@app.errorhandler(404)
def FUN_404(error):
    return render_template("error.html")

@app.errorhandler(405)
def FUN_405(error):
    return render_template("error.html")

@app.errorhandler(500)
def FUN_500(error):
    return render_template("error.html")


################################################
# Functions for running classifier
################################################

# define a simple data batch
Batch = namedtuple('Batch', ['data'])


def download(url, fname=None, dirname=None, overwrite=False):
    """Download an given URL
    Parameters
    ----------
    url : str
        URL to download
    fname : str, optional
        filename of the downloaded file. If None, then will guess a filename
        from url.
    dirname : str, optional
        output directory name. If None, then guess from fname or use the current
        directory
    overwrite : bool, optional
        Default is false, which means skipping download if the local file
        exists. If true, then download the url to overwrite the local file if
        exists.
    Returns
    -------
    str
        The filename of the downloaded file
    """
    if fname is None:
        fname = url.split('/')[-1]

    if dirname is None:
        dirname = os.path.dirname(fname)
    else:
        fname = os.path.join(dirname, fname)
    if dirname != "":
        if not os.path.exists(dirname):
            try:
                logging.info('create directory %s', dirname)
                os.makedirs(dirname)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise OSError('failed to create ' + dirname)

    if not overwrite and os.path.exists(fname):
        logging.info("%s exists, skipping download", fname)
        return fname

    r = requests.get(url, stream=True)
    assert r.status_code == 200, "failed to open %s" % url
    with open(fname, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
    logging.info("downloaded %s into %s successfully", url, fname)
    return fname
#download("https://raw.githubusercontent.com/XD-DENG/flask-app-for-mxnet-img-classifier/master/static/img/screenshot.png")
def get_image(file_location,is_fixed_size,model_image_size, local=False):
    # users can either 
    # [1] upload a picture (local = True)
    # or
    # [2] provide the image URL (local = False)
    if local == True:
        fname = file_location
        name=fname.split('/')[-1]
        image = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)#Image.open(fname)
        img_OpenCV = cv2.imread(fname)  
        print image.shape[1],image.shape[0]
        if is_fixed_size:  # TODO: When resizing we can use minibatch input.
            #resized_image = image.resize(
            #    tuple(reversed(model_image_size)), Image.BICUBIC)
            resized_image=cv2.resize(image, tuple(reversed(model_image_size)), interpolation = cv2.INTER_CUBIC)
        else:
            # Due to skip connection + max pooling in YOLO_v2, inputs must have
            # width and height as multiples of 32.
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            resized_image = image.resize(new_image_size, Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
            print(image_data.shape)
            
    else:
        fname = download(file_location, dirname="static/img_down")#mx.test_utils.
        print('fname',fname.split('/')[-1])
        name=fname.split('/')[-1]
        image = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
        img_OpenCV=cv2.imread(fname)
        if is_fixed_size:  # TODO: When resizing we can use minibatch input.
            resized_image=cv2.resize(image, tuple(reversed(model_image_size)), interpolation = cv2.INTER_CUBIC)
        else:
            # Due to skip connection + max pooling in YOLO_v2, inputs must have
            # width and height as multiples of 32.
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            resized_image = image.resize(new_image_size, Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
            print(image_data.shape)
    image_data = np.array(resized_image, dtype='float32')

    if image_data is None:
         return None,None,None


    return image_data,image,img_OpenCV,name


graph = K.get_session().graph
global yolo_model
yolo_model = load_model(_model_path)
#sess=K.set_session(_sess)

def yolo_predict(test_path,output_path,model_path='model_data/model.h5',anchors_path='model_data/yolo_anchors.txt',
          classes_path='model_data/coco_classes.txt'
          ,weight_path=None,local=False):
    model_path = model_path
    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
    anchors_path = anchors_path
    classes_path = classes_path
    test_path = test_path
    output_path = output_path
    weight_path = weight_path
    #model_path='model_data/model.h5'
    #yolo_model = load_model(model_path)
    
    if not os.path.exists(output_path):
        #print 'Creating output path {}'.format(output_path)
        os.mkdir(output_path)
    #graph = get_graph()
    with graph.as_default():
        # Get input and output tensors
        sess ,yolo_model= get_session()
       
        #sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

        print(yolo_model,'yolo_modelyolo_modelyolo_modelyolo_modelyolo_model')
        #classes file should one class one line
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]

        #anchors should be separated by ,
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)

        #yolo_model = load_model(model_path)
        if weight_path!=None:
            yolo_model.load_weights(weight_path)

        # Verify model, anchors, and classes are compatible
        num_classes = len(class_names)
        num_anchors = len(anchors)
        # TODO: Assumes dim ordering is channel last
        model_output_channels = yolo_model.layers[-1].output_shape[-1]
        assert model_output_channels == num_anchors * (num_classes + 5), \
            'Mismatch between model and given anchor and class sizes. ' \
            'Specify matching anchors and classes with --anchors_path and ' \
            '--classes_path flags.'
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Check if model is fully convolutional, assuming channel last order.
        model_image_size = yolo_model.layers[0].input_shape[1:3]
        is_fixed_size = model_image_size != (None, None)

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / float(len(class_names)), 1., 1.)
                      for x in range(len(class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        # TODO: Wrap these backend operations with Keras layers.
        score_threshold=0.4
        iou_threshold=0.5
        yolo_outputs = convert_result(yolo_model.output, anchors, len(class_names))
        print('yolo_outputs',yolo_outputs)
        input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = draw_helper(
            yolo_outputs,
            input_image_shape,
            to_threshold=score_threshold,
            iou_threshold=iou_threshold)
        points = []
        image_data,image,img_OpenCV,fname=get_image(test_path,is_fixed_size,model_image_size)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                yolo_model.input: image_data,
                input_image_shape: [image.shape[0], image.shape[1]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for {}'.format(len(out_boxes), test_path))

        font = ImageFont.truetype(
            font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.shape[0] + 0.5).astype('int32'))
        thickness = (image.shape[1] + image.shape[0]) // 300
        image=Image.fromarray(cv2.cvtColor(img_OpenCV, cv2.COLOR_BGR2RGB))
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            # 图像从OpenCV格式转换成PIL格式  
            
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            #the result's origin is in top left
            print(label, (left, top), (right, bottom))
            #if label.split(' ')[0] == 'person' :
            #    points.append((int(left),int(top),int(right),int(bottom)))
            
            points.append((label.split(' ')[0],round(float(label.split(' ')[1]),3)))


            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):

                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        image.save(os.path.join(output_path, fname), quality=90)

    return points



################################################
# Functions for Image Archive
################################################

def FUN_resize_img(filename, resize_proportion = 0.3):
    '''
    FUN_resize_img() will resize the image passed to it as argument to be {resize_proportion} of the original size.
    '''
    img=cv2.imread(filename)
    small_img = cv2.resize(img, (0,0), fx=resize_proportion, fy=resize_proportion)
    cv2.imwrite(filename, small_img)

################################################
# Functions Building Endpoints
################################################

@app.route("/", methods = ['POST', "GET"])
def FUN_root():
    # Run correspoing code when the user provides the image url
    # If user chooses to upload an image instead, endpoint "/upload_image" will be invoked
    if request.method == "POST":
        img_url = request.form.get("img_url")
        print('img_url',img_url)
        prediction_result = yolo_predict(img_url,"static/img_pool")
        print prediction_result
        return render_template("index.html", img_src = img_url, prediction_result = prediction_result)
    else:
        return render_template("index.html")


@app.route("/about/")
def FUN_about():
    return render_template("about.html")


ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'bmp']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload_image", methods = ['POST'])
def FUN_upload_image():

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return(redirect(url_for("FUN_root")))
        file = request.files['file']

        # if user does not select file, browser also submit a empty part without filename
        if file.filename == '':
            return(redirect(url_for("FUN_root")))

        if file and allowed_file(file.filename):
            filename = os.path.join("static/img_down", hashlib.sha256(str(datetime.datetime.now())).hexdigest() + secure_filename(file.filename).lower())
            file.save(filename)
            prediction_result = yolo_predict(filename,"static/img_pool",local=True)
            FUN_resize_img(filename)
            return render_template("index.html", img_src = filename, prediction_result = prediction_result)
    return(redirect(url_for("FUN_root")))


################################################
# Start the service
################################################
if __name__ == "__main__":
    #init_model()
    app.run(debug=True,port=9001)