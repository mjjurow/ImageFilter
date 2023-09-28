#largely sourced from: https://www.youtube.com/watch?v=r4L38sc_e6Q

from flask import Flask, request, render_template, redirect, session, url_for
import base64
import cv2
from scipy.interpolate import UnivariateSpline
import numpy as np
from flask_session import Session

# Initiate flask app
app = Flask(__name__)
app.secret_key = #secret key here
app.config['SECRET_KEY'] = #secret key here

# Configure in-memory sessions
app.config['SESSION_TYPE'] = 'filesystem' 
app.config['SESSION_FILE_DIR'] = '/tmp/sessions' # Temporary directory for Flask-Session to use.
app.config['SESSION_PERMANENT'] = False  # Set this as per your requirement.
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'sess_'  

Session(app)

def mapping_function(x,y):
    spl = UnivariateSpline(x,y)
    return spl(range(256)) 

def apply_warm(image):
    increase = mapping_function([0,64,128,192,256], [0,70,140,210,256]) # 0 maps to 0, 256 to 256: keeps everything on scale   
    decrease = mapping_function([0,64,128,192,256], [0,40,90,150,256])
    
    red, green, blue = cv2.split(image)
    red = cv2.LUT(red,increase).astype(np.uint8) # 8 bit
    blue = cv2.LUT(blue,decrease).astype(np.uint8)
    image = cv2.merge((red, green, blue))
    return image

def apply_cool(image):
    increase = mapping_function([0,64,128,192,256], [0,70,140,210,256])
    decrease = mapping_function([0,64,128,192,256], [0,40,90,150,256])
    
    red, green, blue = cv2.split(image) 
    red = cv2.LUT(red,decrease).astype(np.uint8) 
    blue = cv2.LUT(blue,increase).astype(np.uint8)
    image = cv2.merge((red, green, blue))
    return image

def ice_cold(image):
    increase = mapping_function([0,64,128,192,256], [0,70,140,210,256])
    decrease = mapping_function([0,64,128,192,256], [0,20,45,75,256])
    
    red, green, blue = cv2.split(image) 
    red = cv2.LUT(red,decrease).astype(np.uint8) 
    blue = cv2.LUT(blue,increase).astype(np.uint8)
    image = cv2.merge((red, green, blue))
    return image

def bigly_green(image):
    increase = mapping_function([0,64,128,192,256], [0,77,154,220,255])
    
    red, green, blue = cv2.split(image) 
    green = cv2.LUT(green,increase).astype(np.uint8)
    image = cv2.merge((red, green, blue))
    return image

def HDR(img):
    image = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        
        # Clear session for every new image upload.
        session.clear()

        image = cv2.imdecode(np.frombuffer(request.files['image'].read(), np.uint8), -1)
        _, buffer = cv2.imencode('.jpg', image)
        uploaded_image = base64.b64encode(buffer).decode('utf-8')
        session['uploaded_image'] = uploaded_image
        return render_template('index.html', uploaded_image=uploaded_image)

    # Displaying the index page with uploaded and filtered images.
    return render_template('index.html', uploaded_image=session.get('uploaded_image'), filtered_image=session.get('filtered_image'))


@app.route('/apply_filter', methods=['POST'])
def apply_filter():
    filter_name = request.form.get('filter')
    image_data = session.get('uploaded_image')
    if not image_data:
        return redirect(url_for('index'))

    image_nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    image = cv2.imdecode(image_nparr, cv2.IMREAD_COLOR)

    if filter_name == 'Warm':
        filtered_image = apply_warm(image)
    elif filter_name == 'Cool':
        filtered_image = apply_cool(image)
    elif filter_name == 'Ice Cold':
        filtered_image = ice_cold(image)
    elif filter_name == 'Bigly Green':
        filtered_image = bigly_green(image)
    elif filter_name == 'HDR':
        filtered_image = HDR(image)
    else:
        filtered_image = image  # Default: no filter

    _, buffer = cv2.imencode('.jpg', filtered_image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    session['filtered_image'] = encoded_image

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)