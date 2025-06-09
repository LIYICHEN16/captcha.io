import torch
from captcha import CRNN

num_classes = 63 + 1
model = CRNN(num_classes)
model.load_state_dict(torch.load('crnn_captcha.pth', map_location='cpu'))
model.eval()

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 這裡要存整個模型，不是 state_dict！
torch.save(quantized_model, 'crnn_captcha_quantized.pth')

print("量化完成！新檔案：crnn_captcha_quantized.pth")
