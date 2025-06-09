import torch
from captcha import CRNN

# 定義你的類別數
num_classes = 63 + 1

# 載入原本的模型
model = CRNN(num_classes)
model.load_state_dict(torch.load('crnn_captcha.pth', map_location='cpu'))
model.eval()

# 動態量化
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 儲存量化後的模型
torch.save(quantized_model.state_dict(), 'crnn_captcha_quantized.pth')

print("量化完成！新檔案：crnn_captcha_quantized.pth")
