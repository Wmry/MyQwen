import xml.etree.ElementTree as ET

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

params = ['DeepSeek-version', 'DeepSeek-model_path', 'DeepSeek-max_new_tokens',
          'DeepSeek-generated', 'LoRA-enabled', 'LoRA-rank', 'LoRA-alpha', 'LoRA-dropout',
          'LoRA-target_modules', 'LoRA-bias', 'Training-learning_rate', 'Training-batch_size',
          'Training-num_epochs', 'Training-max_seq_length', 'Training-gradient_accumulation',
          'Training-warmup_ratio', 'Training-weight_decay', 'Training-lr_scheduler',
          'Training-fp16', 'Training-bf16', 'Training-device']

def get_value(element):
    """根据类型声明解析XML元素值"""
    value_type = element.get('type', 'str')
    text = element.text.strip() if element.text else ""

    if value_type == 'bool':
        return text.lower() in ('true', '1', 'yes')
    elif value_type == 'int':
        return int(text) if text else 0
    elif value_type == 'float':
        return float(text) if text else 0.0
    elif value_type == 'list':
        return [item.strip() for item in text.split(',')] if text else []
    else:  # str and others
        return text
def load_config(filename):
    try:
        tree = ET.parse(filename)
        root = tree.getroot()

        # 提取设置
        base_info = {}
        # 提取LoRA微调参数
        lora_set = {}
        train_set = {}
        path_set = {}
        for param in params:
            name_c, name_t = param.split("-")
            value = get_value(root.find(name_c).find(name_t))
            if name_c == 'DeepSeek':
                base_info[name_t] = value
            elif name_c == 'LoRA':
                lora_set[name_t] = value
            elif name_c == 'Training':
                train_set[name_t] = value
            elif name_c == 'Path':
                path_set[name_t] = value
        return {"base_info":base_info, "lora_set":lora_set, "train_set":train_set, "path_set":path_set }

    except Exception as e:
        print(f"加载配置出错: {str(e)}")
        raise

def load_base_model(elements):

    elements['base_info']['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    if elements['train_set']['fp16']:
        torch_dtype = torch.float16
    elif elements['train_set']['bf16']:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = "auto"

    print(f"Using device: " + elements['base_info']['device'])
    tokenizer_tmp = AutoTokenizer.from_pretrained(elements['base_info']['model_path'])
    model_tmp = AutoModelForCausalLM.from_pretrained(
        elements['base_info']['model_path'],
        device_map=elements['base_info']['device'],
        torch_dtype=torch_dtype
    )

    return tokenizer_tmp, model_tmp

if __name__ == "__main__":
    params = load_config("./params.xml")
    print(params)