import xml.etree.ElementTree as ET

import math
import torch
from datasets import load_dataset
from scipy.constants import value
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2Model, Qwen2DecoderLayer
from Model import KGQwen2Attention, KGQwen2ForCausalLM, KGQwen2Model, KGQwen2DecoderLayer

params = ['DeepSeek-version', 'DeepSeek-model_path', 'DeepSeek-max_new_tokens',
          'DeepSeek-generated', 'LoRA-enabled', 'LoRA-rank', 'LoRA-alpha', 'LoRA-dropout',
          'LoRA-target_modules', 'LoRA-bias', 'Training-learning_rate', 'Training-batch_size',
          'Training-num_epochs', 'Training-max_seq_length', 'Training-gradient_accumulation',
          'Training-warmup_ratio', 'Training-weight_decay', 'Training-lr_scheduler',
          'Training-fp16', 'Training-bf16', 'Training-device','Paths-train_data','Paths-txtfile_name',
          'Paths-output_dir', 'Paths-logging_dir', 'Paths-checkpoint_dir']

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


def print_trainable_parameters(model: torch.nn.Module):
    """
    打印模型中可训练参数的信息
    """
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            print(f"[Trainable] {name} | shape={tuple(param.shape)} | params={param.numel()}")
            trainable_params += param.numel()

    print("=" * 80)
    print(f"Total trainable params: {trainable_params:,}")
    print(f"Total all params: {all_params:,}")
    print(f"Trainable %: {100 * trainable_params / all_params:.2f}%")
    print("=" * 80)


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

    replace_model(model=model_tmp, elements=elements['base_info'])
    replace_attention_layers(model=model_tmp, elements=elements['base_info'])

    return tokenizer_tmp, model_tmp


def replace_model(model, top_config=None, elements=None, current_depth=0):
    # 如果是顶层调用，保存config
    if top_config is None:
        top_config = model.config

    # 遍历所有子模块
    for name, module in model.named_children():
        if isinstance(module, Qwen2Model):
            new_model = KGQwen2Model(
                top_config,  # 使用顶层config而不是当前模块的config
            ).to(device=elements['device'], dtype=top_config.torch_dtype)
            # 复制原始权重
            model_state_dict = module.state_dict()
            new_model_state_dict = new_model.state_dict()

            # 只复制匹配的键
            for key in model_state_dict:
                if key in new_model_state_dict:
                    new_model_state_dict[key].copy_(model_state_dict[key])

            # 将新层设置为评估模式（如果需要）
            new_model.eval()

            setattr(model, name, new_model)


def replace_attention_layers(model, top_config=None, elements=None, current_depth=0, parent_modle=None):
    """
    递归替换模型中的Qwen2DecoderLayer层，仅替换第一层和中间层

    Args:
        model: 要处理的模型
        top_config: 顶层模型的config对象，用于递归传递
        elements: 包含设备等信息的字典
        current_depth: 当前递归深度，用于跟踪处理进度
    """
    # 如果是顶层调用，保存config
    if top_config is None:
        top_config = model.config

    # 初始化层计数器（仅在顶层调用时）
    if current_depth == 0:
        replace_attention_layers.layer_counter = 0
        replace_attention_layers.total_layers = top_config.num_hidden_layers

    # 遍历所有子模块
    for name, module in model.named_children():
        if isinstance(module, Qwen2DecoderLayer):
            # 获取当前层索引
            layer_idx = replace_attention_layers.layer_counter
            replace_attention_layers.layer_counter += 1

            # 判断是否为第一层或中间层
            is_first_layer = (layer_idx == 0)
            is_middle_layer = (layer_idx == replace_attention_layers.total_layers // 2)

            if is_first_layer or is_middle_layer:
                print(f"Replacing layer {layer_idx} with KGQwen2DecoderLayer")

                # 创建知识增强版解码器层，使用顶层config
                new_layer = KGQwen2DecoderLayer(
                    top_config,  # 使用顶层config而不是当前模块的config
                    embed_tokens=parent_modle.embed_tokens,  # 共享顶层的embedding
                    weight_k=parent_modle.W_k,
                    weight_q=parent_modle.W_q,
                    layer_idx=layer_idx
                ).to(device=elements['device'], dtype=top_config.torch_dtype)

                # 复制原始权重
                original_state_dict = module.state_dict()
                new_state_dict = new_layer.state_dict()

                # 只复制匹配的键
                for key in original_state_dict:
                    if key in new_state_dict:
                        new_state_dict[key].copy_(original_state_dict[key])

                # 将新层设置为评估模式（如果需要）
                new_layer.eval()

                setattr(model, name, new_layer)
        else:
            # 递归处理子模块，传递顶层config
            replace_attention_layers(module, top_config, elements, current_depth + 1, model)

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

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of = 8  # 可选：提高GPU效率
    )

    # 7. 转为 PyTorch DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,  # 样本数
        shuffle=True,
        collate_fn=data_collator
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )
    return train_dataset, test_dataset

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling


def load_my_dataset_hugging_face_method(txt_path, txt_name="TestKG.txt", tokenizer=None, target_multiple=32):
    """
    加载并预处理文本数据集，确保数据集大小是target_multiple的倍数

    参数:
    - txt_path: 文本文件路径
    - txt_name: 文本文件名
    - tokenizer: 分词器，如果为None则自动加载
    - target_multiple: 目标倍数，确保数据集大小是该值的倍数
    """
    # 1. 加载本地txt文本，每行一个样本
    dataset = load_dataset(
        "text",
        data_files={"raw": f"{txt_path}/{txt_name}"},
        encoding="GB18030"
    )

    # 2. 过滤空行和无效行
    def filter_empty_lines(example):
        stripped = example["text"].strip()
        return stripped != "" and not stripped.isspace() and len(stripped) > 1

    dataset = dataset["raw"].filter(filter_empty_lines)

    # 计算调整后的数据集大小
    def adjust_to_multiple(size, multiple):
        """调整大小到最接近的multiple的倍数"""
        return math.floor(size / multiple) * multiple

    # 3. 划分训练/验证/测试集 (80% train, 10% valid, 10% test)
    # 首先获取原始大小
    total_size = len(dataset)
    train_size = adjust_to_multiple(total_size * 0.8, target_multiple)
    valid_test_size = adjust_to_multiple(total_size * 0.0025, target_multiple)

    # 使用select方法选择指定数量的样本
    train_dataset = dataset.select(range(train_size))
    remaining = dataset.select(range(train_size, total_size))

    # 划分验证集和测试集
    valid_dataset = remaining.select(range(valid_test_size))
    test_dataset = remaining.select(range(valid_test_size, valid_test_size * 80))

    print(f"数据集大小调整: 总样本 {total_size} -> 训练集 {len(train_dataset)}, "
          f"验证集 {len(valid_dataset)}, 测试集 {len(test_dataset)}")

    # 确保所有数据集大小都是target_multiple的倍数
    assert len(train_dataset) % target_multiple == 0, "训练集大小不是目标倍数"
    assert len(valid_dataset) % target_multiple == 0, "验证集大小不是目标倍数"
    assert len(test_dataset) % target_multiple == 0, "测试集大小不是目标倍数"

    # 4. 初始化分词器
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            "D:/Users/xiangyu/download/Qwen_1.5B_Baseline",
            trust_remote_code=True
        )

    # 添加填充token（如果尚未添加）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 5. 定义分词函数
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,  # 添加最大长度限制以防止内存问题
            # 不在此处填充，由data_collator处理
        )

    # 6. 对三个数据集做token化
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    valid_dataset = valid_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 7. 定义collator，自动生成labels并动态padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 自回归语言模型，不是MLM
        pad_to_multiple_of=8  # 填充到8的倍数，有助于某些硬件加速
    )

    return train_dataset, valid_dataset, test_dataset, data_collator


if __name__ == "__main__":
    import torch

    print("CUDA:", torch.version.cuda)
    print("Torch:", torch.__version__)
    print("Device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0))

    capability = torch.cuda.get_device_capability(0)
    print("Compute Capability:", capability)

    params = load_config("./params.xml")
    # a, b = load_my_dataset("D:/Users/xiangyu")
    print(params)