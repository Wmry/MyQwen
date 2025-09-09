import xml.etree.ElementTree as ET
import torch
from datasets import load_dataset
from scipy.constants import value
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2Model
from Model import KGQwen2Attention, KGQwen2ForCausalLM, KGQwen2Model

params = ['DeepSeek-version', 'DeepSeek-model_path', 'DeepSeek-max_new_tokens',
          'DeepSeek-generated', 'LoRA-enabled', 'LoRA-rank', 'LoRA-alpha', 'LoRA-dropout',
          'LoRA-target_modules', 'LoRA-bias', 'Training-learning_rate', 'Training-batch_size',
          'Training-num_epochs', 'Training-max_seq_length', 'Training-gradient_accumulation',
          'Training-warmup_ratio', 'Training-weight_decay', 'Training-lr_scheduler',
          'Training-fp16', 'Training-bf16', 'Training-device','Paths-train_data','Paths-txtfile_name']

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
            elif name_c == 'Paths':
                path_set[name_t] = value
        return {"base_info":base_info, "lora_set":lora_set, "train_set":train_set, "path_set":path_set }

    except Exception as e:
        print(f"加载配置出错: {str(e)}请检查params文件是否配置齐全")
        raise

def load_base_model(elements):

    elements['base_info']['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    if elements['train_set']['fp16']:
        torch_dtype = torch.float16
    elif elements['train_set']['bf16']:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = "auto" # 建议不要用 "auto"，显式更好

    print(f"Using device: " + elements['base_info']['device'])
    tokenizer_tmp = AutoTokenizer.from_pretrained(elements['base_info']['model_path'])
    model_tmp = AutoModelForCausalLM.from_pretrained(
        elements['base_info']['model_path'],
        device_map=elements['base_info']['device'],
        torch_dtype=torch_dtype
    )

    replace_attention_layers(model=model_tmp, elements=elements['base_info'])

    return tokenizer_tmp, model_tmp


def replace_attention_layers(model, top_config=None, elements=None, current_depth=0):
    """
    递归替换模型中的Qwen2Attention层，仅替换第一层和中间层

    Args:
        model: 要处理的模型
        top_config: 顶层模型的config对象，用于递归传递
        elements: 包含设备等信息的字典
        current_depth: 当前递归深度，用于跟踪处理进度
    """
    # 如果是顶层调用，保存config
    if top_config is None:
        top_config = model.config

    # 遍历所有子模块
    for name, module in model.named_children():
        if isinstance(module, Qwen2Model):
            new_attn = KGQwen2Model(
                top_config,  # 使用顶层config而不是当前模块的config
            ).to(device=elements['device'], dtype=top_config.torch_dtype)
            # 复制原始权重
            attn_state_dict = module.state_dict()
            new_attn_state_dict = new_attn.state_dict()

            # 只复制匹配的键
            for key in attn_state_dict:
                if key in new_attn_state_dict:
                    new_attn_state_dict[key].copy_(attn_state_dict[key])

            # 将新层设置为评估模式（如果需要）
            new_attn.eval()

            setattr(model, name, new_attn)

        # if isinstance(module, Qwen2Attention):
            # # 检查是否是第一层或中间层
            # if hasattr(module, 'layer_idx'):
            #     layer_idx = module.layer_idx
            #     # 判断是否为第一层或中间层
            #     is_first_layer = (layer_idx == 0)
            #     is_middle_layer = (layer_idx == top_config.num_hidden_layers // 2)
            #
            #     if is_first_layer or is_middle_layer:
            #         print(f"Replacing layer {layer_idx} with KGQwen2Attention")
            #
            #         # 创建知识增强版注意力层，使用顶层config
            #         new_attn = KGQwen2Attention(
            #             top_config,  # 使用顶层config而不是当前模块的config
            #             layer_idx=layer_idx
            #         ).to(device=elements['device'], dtype=top_config.torch_dtype)
            #
            #         # 复制原始权重
            #         attn_state_dict = module.state_dict()
            #         new_attn_state_dict = new_attn.state_dict()
            #
            #         # 只复制匹配的键
            #         for key in attn_state_dict:
            #             if key in new_attn_state_dict:
            #                 new_attn_state_dict[key].copy_(attn_state_dict[key])
            #
            #         # 将新层设置为评估模式（如果需要）
            #         new_attn.eval()
            #
            #         setattr(model, name, new_attn)
            # else:
            #     print(f"Warning: Qwen2Attention module '{name}' has no layer_idx attribute")
        else:
            # 递归处理子模块，传递顶层config
            replace_attention_layers(module, top_config, elements, current_depth + 1)

# 针对 inputs 做智能处理：input_ids 保持 long，其它可以变 dtype
def smart_to_dtype_and_device(inputs, model_dtype, device):
    new_inputs = {}
    for k, v in inputs.items():
        if v.dtype in [torch.long, torch.int]:  # 不能改 dtype 的情况
            new_inputs[k] = v.to(device)
        else:
            new_inputs[k] = v.to(dtype=model_dtype, device=device)
    return new_inputs

def load_my_dataset(txt_path, txt_name="TestKG.txt",  tokenizer=None, batch_size=2):
    # 1. 加载本地txt文本，每行一个样本
    dataset = load_dataset(txt_path, data_files=txt_name, encoding='GB18030')

    # 2. 过滤空行和无效行
    def filter_empty_lines(example):
        # 去除首尾空白后检查是否为空
        stripped = example['text'].strip()
        # 排除空行和只有标点符号的行
        return stripped != '' and not stripped.isspace() and len(stripped) > 1

    dataset = dataset.filter(filter_empty_lines)

    # 2. 切分训练集/测试集
    dataset_split = dataset["train"].train_test_split(test_size=0.1, seed=42)

    train_dataset = dataset_split["train"]
    test_dataset = dataset_split["test"]

    # 3. 初始化分词器（以Qwen为例）
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("D:/Users/xiangyu/download/Qwen_1.5B_Baseline", trust_remote_code=True)

    # 4. 定义分词函数（动态截断）
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

    # 5. 对训练集和测试集做token化
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # 删除原始文本列
    train_dataset = train_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])

    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer,
    #     mlm=False,
    #     pad_to_multiple_of = 8  # 可选：提高GPU效率
    # )
    #
    # # 7. 转为 PyTorch DataLoader
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,  # 样本数
    #     shuffle=True,
    #     collate_fn=data_collator
    # )
    #
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     collate_fn=data_collator
    # )
    return train_dataset, test_dataset

if __name__ == "__main__":
    params = load_config("./params.xml")
    # a, b = load_my_dataset("D:/Users/xiangyu")
    print(params)