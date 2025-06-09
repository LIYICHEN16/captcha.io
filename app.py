from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
from captcha import CRNN, idx_to_char
from torchvision import transforms
from torch.nn.modules.container import Sequential

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

num_classes = 63 + 1

# 這一行一定要放在 torch.load 前
torch.serialization.add_safe_globals([CRNN])

model = torch.load('crnn_captcha_quantized.pth', map_location='cpu', weights_only=False)
model.eval()

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
