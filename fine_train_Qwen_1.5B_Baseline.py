from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from My_Unit import load_config, load_base_model, smart_to_dtype_and_device, load_my_dataset

def load_model(elements):
    tokenizer_tmp, model_tmp = load_base_model(elements)
    return tokenizer_tmp, model_tmp

def prepare_inputs(inputs, model):
    # 用 next(model.parameters()) 拿到实际 dtype 和 device
    param = next(model.parameters())
    device = param.device
    dtype = param.dtype

    return {
        k: v.to(device) if v.dtype in (torch.long, torch.int) else v.to(dtype=dtype, device=device)
        for k, v in inputs.items()
    }


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

def run(model, dataloader, tokenizer):
    model.train()
    for data in dataloader:
        print(data["attention_mask"].sum(dim=-1))



# def run(model, tokenizer, input_text):
#     model.eval()
#     inputs = tokenizer(input_text, return_tensors="pt")
#
#     # 保证输入和模型 device/dtype 匹配
#     inputs = prepare_inputs(inputs, model)
#
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=5000,
#             pad_token_id=tokenizer.eos_token_id
#         )
#
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print(generated_text)


if __name__ == "__main__":
    params = load_config("./params.xml")
    tokenizer, model = load_model(params)
    train_path = params['path_set']['train_data']
    train_txtfile = params['path_set']['txtfile_name']
    train_loader, test_loader = load_my_dataset(txt_path=train_path, txt_name=train_txtfile, tokenizer=tokenizer)
    run(model, train_loader, tokenizer)
    # run(model, tokenizer, "请给出一篇500字的自我介绍，介绍大模型计算和编程")
    # print(type(model))