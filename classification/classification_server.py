from datetime import timedelta
from flask import Flask, jsonify, request, current_app, make_response, send_file
from flask_cors import CORS
from functools import update_wrapper
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.transform import resize
import tensorflow as tf


# configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})


DATADIR = r"C:\Users\carte\Desktop\BeeMachine\Bombus_images\Bumble_iNat_BugGuide_BBW"
CATEGORIES = ["Bombus_affinis", "Bombus_appositus", "Bombus_auricomus", "Bombus_bifarius", "Bombus_bimaculatus", "Bombus_borealis", 
              "Bombus_caliginosus", "Bombus_centralis", "Bombus_citrinus", "Bombus_crotchii", "Bombus_cryptarum", "Bombus_fernaldae_flavidus",
              "Bombus_fervidus", "Bombus_flavifrons", "Bombus_fraternus", "Bombus_frigidus", "Bombus_griseocollis", "Bombus_huntii", 
              "Bombus_impatiens", "Bombus_insularis", "Bombus_melanopygus", "Bombus_mixtus", "Bombus_morrisoni", "Bombus_nevadensis", 
              "Bombus_occidentalis", "Bombus_pensylvanicus_sonorus", "Bombus_perplexus", "Bombus_rufocinctus", "Bombus_sandersoni", 
              "Bombus_sitkensis", "Bombus_sylvicola", "Bombus_ternarius", "Bombus_terricola", "Bombus_vagans", "Bombus_vandykei", "Bombus_vosnesenskii"]
NUM_CLASSES = len(CATEGORIES) #Number of classes (e.g., species)
IMG_SIZE = 299                #length and width of input images
UPLOAD_PATH = '/mnt/c/Users/carte/Desktop/SeniorDesignProject/server/uploads'

MODEL = load_model('/mnt/c/Users/carte/Desktop/BeeMachine/model_InceptionV3_04-12-2020.h5')


def crossdomain(origin=None, methods=None, headers=None, max_age=21600,
                attach_to_all=True, automatic_options=True):
    """Decorator function that allows crossdomain requests.
      Courtesy of
      https://blog.skyred.fi/articles/better-crossdomain-snippet-for-flask.html
    """
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    # use str instead of basestring if using Python 3.x
    if headers is not None and not isinstance(headers, str):
        headers = ', '.join(x.upper() for x in headers)
    # use str instead of basestring if using Python 3.x
    if not isinstance(origin, str):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        """ Determines which methods are allowed
        """
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        """The decorator function
        """
        def wrapped_function(*args, **kwargs):
            """Caries out the actual cross domain code
            """
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers
            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            h['Access-Control-Allow-Credentials'] = 'true'
            h['Access-Control-Allow-Headers'] = \
                "Origin, X-Requested-With, Content-Type, Accept, Authorization"
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator


def get_classification(path):
    img_array = plt.imread(path)
    img_array = resize(img_array, (IMG_SIZE, IMG_SIZE,3))
    img_array = np.array([img_array,])
    model_out = MODEL.predict(img_array)[0,:]
    probabilities = np.argsort(model_out)
    return CATEGORIES[probabilities[-1]]


@app.route('/classify', methods=['POST'])
@crossdomain(origin='*')
def classify():
    response_object = {'status': 'success'}
    if request.method == 'POST':
        data = request.json
        filename = data['filename']
        path = os.path.join(UPLOAD_PATH, filename)
        classification = get_classification(path)
        response_object['message'] = classification
    return jsonify(response_object)


if __name__ == '__main__':
    app.run(port=6969, use_reloader=False, threaded=False)
