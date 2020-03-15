from datetime import timedelta
from flask import Flask, jsonify, request, current_app, make_response, send_file
from flask_cors import CORS
from functools import update_wrapper
import os
import time
from werkzeug.datastructures import ImmutableMultiDict
from werkzeug.utils import secure_filename


# configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

IMAGES = [
    {
        'name': 'test.png',
        'path': '/blah/blah/test.png',
        'time': time.time()
    }
]


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


@app.route('/images', methods=['GET', 'POST'])
@crossdomain(origin='*')
def all_images():
    response_object = {'status': 'success'}
    if request.method == 'POST':
        files = dict(request.files)
        for file in files:
            filename = secure_filename(files[file].filename)
            # files[file].save(os.path.join('uploads', filename))             Unsave once ready.
            IMAGES.append({
                'name': filename,
                'path': os.path.join('uploads', filename),
                'time': time.time()
            })
        response_object['message'] = 'Image added!'
    else:
        response_object['images'] = IMAGES
    return jsonify(response_object)


# @app.route('/get_image')
# def get_image():
#     if request.args.get('type') == '1':
#        filename = 'ok.gif'
#     else:
#        filename = 'error.gif'
#     return send_file(filename, mimetype='image/gif')


# @app.route('/prediction/<filename>')
# def prediction(filename):
#     #Step 1
#     my_image = plt.imread(os.path.join('uploads', filename))
#     #Step 2
#     my_image_re = resize(my_image, (32,32,3))
    
#     #Step 3
#     with graph.as_default():
#       set_session(sess)
#       probabilities = model.predict(np.array( [my_image_re,] ))[0,:]
#       print(probabilities)
# #Step 4
#       number_to_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#       index = np.argsort(probabilities)
#       predictions = {
#         "class1":number_to_class[index[9]],
#         "class2":number_to_class[index[8]],
#         "class3":number_to_class[index[7]],
#         "prob1":probabilities[index[9]],
#         "prob2":probabilities[index[8]],
#         "prob3":probabilities[index[7]],
#       }
# #Step 5
#     return render_template('predict.html', predictions=predictions)


if __name__ == '__main__':
    app.run()
    