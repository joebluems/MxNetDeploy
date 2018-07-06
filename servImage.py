import os, urllib
from flask import Flask, request, Response
import numpy as np
import cv2
import mxnet as mx
import pandas as pd
import json
from collections import namedtuple
from flask import Flask, jsonify, request

##################
### READ MODEL ###
##################
def download(url,prefix=''):
    filename = prefix+url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.urlretrieve(url, filename)

download('/mapr/my.cluster/user/mapr/containers/mxnet/model/resnet-152-symbol.json', 'full-')
download('/mapr/my.cluster/user/mapr/containers/mxnet/model/resnet-152-0000.params', 'full-')
download('/mapr/my.cluster/user/mapr/containers/mxnet/model/synset.txt', 'full-')

with open('full-synset.txt', 'r') as f:
    synsets = [l.rstrip() for l in f]

sym, arg_params, aux_params = mx.model.load_checkpoint('full-resnet-152', 0)

mod = mx.mod.Module(symbol=sym, context=mx.cpu())
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
mod.set_params(arg_params, aux_params)


#######################
###HELPER FUNCTION ####
#######################
Batch = namedtuple('Batch', ['data'])

def predict(img, mod, synsets):
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]

    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    classes = []
    for i in a[0:5]:
        classes.append(synsets[i])

    output = "{'p1':'%s (%.3f)','p2':'%s (%.3f)','p3':'%s (%.3f)'}" \
       % (classes[0],prob[a[0]],classes[1],prob[a[1]],classes[2],prob[a[2]])

    return output



# Initialize the Flask application
app = Flask(__name__)

# route http posts to this method
@app.route('/api/test', methods=['POST'])
def test():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # classify images  & return json
    predictions = predict(img,mod, synsets)
    responses = jsonify(predictions)
    responses.status_code = 200
    return (responses)


