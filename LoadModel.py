from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_path = "/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-7B"  # 修改为你的模型目录路径

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device,  
    torch_dtype="auto"
).cuda()
names = ['']

operation = torch.optim.Adam(model.parameters(), lr=1e-5)

text = "请您计算67*89和88乘以167并给出解答过程"
inputs = tokenizer(text, return_tensors="pt").to(device)

# 修正后代码：
with torch.no_grad():

    generated_ids = model.generate(**inputs, max_new_tokens=5000, pad_token_id=tokenizer.eos_token_id)
# 解码生成的token（跳过特殊令牌）
generated_text = tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True)

print("生成结果:", generated_text)
