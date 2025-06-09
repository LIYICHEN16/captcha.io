# app.py
from flask import Flask, request, jsonify
import torch
from PIL import Image
from flask_cors import CORS
from captcha import CRNN, idx_to_char

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# 下面程式碼同你原本

if __name__ == '__main__':
    app.run(debug=True)
 # 你的模型與字典

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# 啟動時載入模型
num_classes = 63 + 1
model = CRNN(num_classes)
model.load_state_dict(torch.load('crnn_captcha.pth', map_location='cpu'))  # 指定模型檔名
model.eval()

from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((40, 150)),
    transforms.ToTensor(),
])

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    img = Image.open(image_file.stream).convert('L')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(2)[0].cpu().numpy()
        seq_str = ''
        prev = -1
        for idx in pred:
            if idx != prev and idx != 0:
                seq_str += idx_to_char.get(idx, '')
            prev = idx

    return jsonify({'result': seq_str})

if __name__ == '__main__':
    app.run(debug=True)

