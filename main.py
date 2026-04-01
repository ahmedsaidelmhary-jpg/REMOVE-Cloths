import os
import uuid
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder='.')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

def remove_clothes_simple(image_path):
    # فتح الصورة
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # تحويل إلى HSV
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # كشف الجلد
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # كشف الملابس
    lower_clothes = np.array([0, 0, 0])
    upper_clothes = np.array([180, 255, 100])
    clothes_mask = cv2.inRange(hsv, lower_clothes, upper_clothes)
    
    # استبعاد الوجه
    h, w = img_array.shape[:2]
    face_region = int(h * 0.25)
    clothes_mask[:face_region, :] = 0
    
    # تنعيم
    kernel = np.ones((5,5), np.uint8)
    clothes_mask = cv2.dilate(clothes_mask, kernel, iterations=1)
    clothes_mask = cv2.GaussianBlur(clothes_mask, (5,5), 0)
    
    # ماسك 3 قنوات
    mask_3ch = np.stack([clothes_mask] * 3, axis=-1) / 255.0
    
    # لون جلدي
    skin_color = np.array([245, 215, 185])
    
    result = img_array.copy()
    for c in range(3):
        result[:,:,c] = np.where(mask_3ch[:,:,c] > 0.5, skin_color[c], result[:,:,c])
    
    result = Image.fromarray(result)
    return result

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file'}), 400
    
    filename = str(uuid.uuid4()) + '.jpg'
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(original_path)
    
    result_image = remove_clothes_simple(original_path)
    
    result_filename = 'result_' + filename
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    result_image.save(result_path)
    
    return jsonify({
        'success': True,
        'result_url': f'/result/{result_filename}'
    })

@app.route('/result/<filename>')
def get_result(filename):
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)