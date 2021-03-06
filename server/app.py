from datetime import timedelta
from flask import Flask, jsonify, request, current_app, make_response, send_file
from flask_cors import CORS
from functools import update_wrapper
from json import dumps
import os
from requests import post
import time
from werkzeug.datastructures import ImmutableMultiDict
from werkzeug.utils import secure_filename


# configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}}, expose_headers=["x-suggested-filename", "Content-Disposition"])

UPLOAD_PATH = '/mnt/c/Users/carte/Desktop/SeniorDesignProject/server/uploads'
POST_URL = 'http://127.0.0.1:6969/classify'
headers = {'content-type': 'application/json'}

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


@app.route('/images', methods=['POST'])
@crossdomain(origin='*')
def all_images():
    response_object = {'status': 'success'}
    if request.method == 'POST':
        files = dict(request.files)
        for file in files:
            filename = time.strftime("%Y%m%d-%H%M%S.jpg")            # NOTES, may need to remove the index 0 on unix systems.
            files[file][0].save(os.path.join('uploads', filename))             # Unsave once ready.
            path = os.path.join(UPLOAD_PATH, filename)
            # IMAGES.append({
            #     'name': filename,
            #     'path': os.path.join('uploads', filename),
            #     'time': time.time()
            # })
            my_data = {'filename': filename}
            response = post(POST_URL, data=dumps(my_data), headers=headers)
            print(response.text)
            response = response.json()['message'].split('_')
            message = response[0] + ' ' + response[1].capitalize()
            new_filename = message + '.jpg'
        # response_object['message'] = message
        response_object = send_file(path, mimetype='image/jpg', as_attachment=True, attachment_filename=new_filename, conditional=False)
        response_object.headers['x-suggested-filename'] = new_filename
        response_object.headers["Access-Control-Expose-Headers"] = 'x-suggested-filename'
    # return jsonify(response_object)
    return response_object


if __name__ == '__main__':
    app.run()
    