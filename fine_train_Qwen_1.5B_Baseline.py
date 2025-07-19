from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from My_Unit import load_config, load_base_model

def load_model(elements):
    tokenizer_tmp, model_tmp = load_base_model(elements)
    return tokenizer_tmp, model_tmp

def train(data_tmp, tokenizer_tmp, model_tmp, epochs, is_eval=False):
    if is_eval:
        model_tmp.eval()

    # for epoch in range(epochs):
    #     opt.zero_grad()
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, norm_type=2)
    #     opt.step()
    pass

def test(data_tmp, tokenizer_tmp, model_tmp, epochs):
    model_tmp.eval()
 
    # for epoch in range(epochs):
    #     # opt.zero_grad()
    #     # loss.backward()
    #     # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, norm_type=2)
    #     # opt.step()
    pass

def run(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # 修正后代码：
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=5000, pad_token_id=tokenizer.eos_token_id)
    # 解码生成的token（跳过特殊令牌）
    generated_text = tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True)
    print(generated_text)



if __name__ == "__main__":
    params = load_config("./params.xml")
    tokenizer, model = load_model(params)
    run(model, tokenizer, "0 个文件已提交，2 个文件提交失败: 调整文件 Committer identity unknown  *** Please tell me who you are.  Run  git config --global user.email ‘you@example.com’ git config --global user.name ‘Your Name’  to set your account's default identity. Omit --global to set the identity only in this repository.  unable to auto-detect email address (got 'qiqi@DESKTOP-TA8M4CB.(none)')"
                          "问题如何解决")
    # print(type(model))