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

if __name__ == "__main__":
    params = load_config("./params.xml")
    tokenizer, model = load_model(params)
    print(type(model))