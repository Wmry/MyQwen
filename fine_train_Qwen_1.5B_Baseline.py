import logging

from safetensors.torch import save_model
from scipy.ndimage import label
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, PreTrainedModel, get_scheduler
import torch
from Model import KGQwen2Attention, KGQwen2DecoderLayer
import torch.optim as optim
from My_Unit import load_config, load_base_model, smart_to_dtype_and_device, load_my_dataset, \
    load_my_dataset_hugging_face_method, print_trainable_parameters
import math
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
import os
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================
# 2. 评价指标（PPL）
# =========================
total_loss_accum = 0.0
total_tokens_accum = 0

# 配置日志
logging.basicConfig(
    filename='training.log',
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(elements):
    tokenizer_tmp, model_tmp = load_base_model(elements)
    return tokenizer_tmp, model_tmp


def prepare_inputs(inputs, model):
    # 用 next(model.parameters()) 拿到实际 dtype 和 device
    param = next(model.parameters())
    device = param.device
    dtype = param.dtype
    print("model device: ", device)

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


def apply_lora(model_tmp: PreTrainedModel):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=1,
        lora_alpha=1,
        lora_dropout=0.1,
        rank_pattern={
            "encode_relation.W_q":16,
            "encode_relation.W_k":16,
            "encode_relation.W_v":16,
            "encode_relation.update":16,
            "model.relation_W_k": 16,
            "model.relation_W_q": 16,
            "lm_head": 1,
        },
        alpha_pattern={
            "encode_relation.W_q":64,
            "encode_relation.W_k":64,
            "encode_relation.W_v":64,
            "encode_relation.update":64,
            "model.relation_W_k": 64,
            "model.relation_W_q": 64,
            "lm_head": 1,
        },
        target_modules=[
            # KGQwen2DecoderLayer中的encode_relation相关参数
            "encode_relation.W_q",
            "encode_relation.W_k",
            "encode_relation.W_v",
            "encode_relation.update",
            "model.relation_W_k",
            "model.relation_W_q",
            "lm_head",
        ],
        # 指定需要训练的基础模型层
        modules_to_save=[
            "encode_relation.W_q",
            "encode_relation.W_k",
            "model.relation_W_q",
            "model.relation_W_k",
            "encode_relation.W_v",
            "encode_relation.update"
        ]
        # modules_to_save=[]
    )

    model_tmp = get_peft_model(model_tmp, lora_config)
    model_tmp.print_trainable_parameters()
    return model_tmp


def compute_metrics(eval_pred):
    """
    按 batch 累积指标，避免一次性保存全量 logits
    """
    global total_loss_accum, total_tokens_accum

    logits, labels = eval_pred

    # 转 torch
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    # shift
    shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = labels[..., 1:].reshape(-1)

    # mask掉 ignore_index
    mask = shift_labels != -100
    masked_logits = shift_logits[mask]
    masked_labels = shift_labels[mask]

    # batch loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')  # sum 而不是 mean
    batch_loss = loss_fct(masked_logits.to(torch.float32), masked_labels.to(torch.long))

    # 累积
    total_loss_accum += batch_loss.item()
    total_tokens_accum += masked_labels.numel()

    perplexity = math.exp(batch_loss.item() / masked_labels.numel())

    return {"perplexity": perplexity}


def run(total_loss_accum, total_tokens_accum):
    # =========================
    # 加载模型与数据
    # =========================
    params = load_config("./params.xml")
    tokenizer, model = load_model(params)

    train_path = params['path_set']['train_data']
    train_txtfile = params['path_set']['txtfile_name']
    model_output_path = params['path_set']['output_dir']
    model_train_log = params['path_set']['logging_dir']
    checkpoint_dir = params['path_set']['checkpoint_dir']

    train_dataset, valid_dataset, data_collator = load_my_dataset_hugging_face_method(
        txt_path=train_path,
        txt_name=train_txtfile,
        tokenizer=tokenizer,
        target_multiple=256
    )

    train_length = len(train_dataset)
    train_batch_size = 16

    # =========================
    # 应用 LoRA
    # =========================
    model = apply_lora(model)
    model.config.use_cache = False
    print_trainable_parameters(model)

    # =========================
    # 优化器参数分组
    # =========================
    relation_params, lm_head_params, other_lora_params = [], [], []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 1. 您的自定义关系模块 - 建议中等偏上的学习率 (因为它可能是新加的核心模块)
        if "encode_relation" in n or "relation_W" in n:
            relation_params.append(p)
        # 2. 输出头 - 建议较高的学习率 (因为它直接负责最终预测，需要快速适应)
        elif "lm_head" in n:
            lm_head_params.append(p)
        # 3. 其他所有LoRA参数 (通常是Q, K, V等) - 使用较低的学习率 (微调预训练知识)
        else:
            other_lora_params.append(p)

    param_optimizer = [
        {"params": relation_params, "lr": 3.05e-4, "weight_decay": 0.01},  # 新增关系模块，中等LR
        {"params": lm_head_params, "lr": 2e-5, "weight_decay": 0.0},  # 输出头，最高LR
        {"params": other_lora_params, "lr": 2e-5, "weight_decay": 0.01},  # 其他LoRA参数，保守LR
    ]

    optimizer = AdamW(param_optimizer, betas=(0.9, 0.999), eps=1e-8)
    # optimizer = AdamW(model.parameters(), lr=3e-4)

    # optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    num_training_steps = 4832
    num_warmup_steps = int(num_training_steps * 0.02)  # 2% warmup

    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    # =========================
    # 开启 Gradient Checkpointing
    # =========================
    model.gradient_checkpointing_enable()
    # =========================
    # 训练参数
    # =========================
    training_args = TrainingArguments(
        learning_rate= 2.05e-4,
        output_dir=checkpoint_dir,  # 输出目录
        save_only_model=True,
        overwrite_output_dir=True,  # 覆盖旧输出
        num_train_epochs=2,  # 训练 epoch
        per_device_train_batch_size=16,  # 训练 batch（可适当调大，看显存）
        per_device_eval_batch_size=1,  # 验证 batch，小一点避免 OOM
        gradient_accumulation_steps=4,  # 累积梯度，相当于扩大 batch

        fp16=False,  # 不用 fp16
        bf16=True,  # 用 bf16（A100/8.9 支持，数值更稳定）
        gradient_checkpointing=True,  # 启用梯度检查点，省显存
        eval_strategy="steps",  # 按 step 验证
        eval_steps=128,  # 验证间隔
        save_steps=256,  # 保存间隔（必须是 eval_steps 的倍数）
        load_best_model_at_end=True,  # 保存最优模型
        metric_for_best_model="loss",  # 以 loss 作为最优标准
        greater_is_better=False,

        save_total_limit=2,  # 最多保留 2 个 checkpoint

        logging_dir=model_train_log,  # 日志
        logging_steps=16,  # 每 64 步记录一次

        # 🔑 避免 eval logits 堆积爆显存
        eval_accumulation_steps=64,  # 每 64 个 batch 把 logits 搬到 CPU
        include_inputs_for_metrics=False,  # 不保存输入到 metrics
        remove_unused_columns=False,  # 减少数据集多余拷贝
        dataloader_num_workers=2,  # 多线程数据加载
        dataloader_pin_memory=True,  # 固定内存加速
    )

    # =========================
    # Trainer
    # =========================
    torch.cuda.empty_cache()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),  # 传入自定义 optimizer
    )

    # =========================
    # 训练并保存
    # =========================
    trainer.train()
    trainer.save_model(model_output_path)

    trainer.evaluate()
    # 用累积 loss 计算 perplexity
    perplexity = math.exp(total_loss_accum / total_tokens_accum)
    print("Validation Perplexity:", perplexity)

    # =========================
    # 绘制曲线
    # =========================
    logs = trainer.state.log_history
    # epochs, ppl, losses = [], [], []
    #
    # for entry in logs:
    #     if "epoch" in entry:
    #         if "eval_perplexity" in entry:
    #             epochs.append(entry["epoch"])
    #             ppl.append(entry["eval_perplexity"])
    #         if "loss" in entry:
    #             losses.append(entry["loss"])
    #
    # plt.figure(figsize=(8, 5))
    # plt.plot(epochs, ppl, marker="o", label="Eval Perplexity")
    # plt.plot(range(len(losses)), losses, linestyle="--", label="Train Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.grid()
    # plt.title("Training & Evaluation Curve")
    # plt.show()


def valid():
    # =========================
    # 加载模型与数据
    # =========================
    params = load_config("./params.xml")
    tokenizer, model_tmp = load_model(params)
    train_path = params['path_set']['train_data']
    train_txtfile = params['path_set']['txtfile_name']
    model_output_path = params['path_set']['output_dir']
    model_train_log = params['path_set']['logging_dir']
    checkpoint_dir = params['path_set']['checkpoint_dir']
    device = params['base_info']['device']
    model_tmp = PeftModel.from_pretrained(model_tmp, model_output_path)

    # 5. 切换到适配器模式（如果需要使用多个适配器）
    model_tmp.set_adapter("default")  # 使用默认适配器
    torch.cuda.empty_cache()
    model_tmp.eval()
    text = "请介绍“村里的其他孩子也是“狗娃”“二蛋”之类的被人一直称呼着”的含义"
    inputs = tokenizer(text, return_tensors="pt", max_length=125, padding=True).to(device)

    with torch.no_grad():
        generated_ids = model_tmp.generate(**inputs, max_new_tokens=5000, pad_token_id=tokenizer.eos_token_id)
    # 解码生成的token（跳过特殊令牌）
    generated_text = tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True)

    print("生成结果:", generated_text)
    # 现在模型已准备好使用
    print("模型加载完成！")

if __name__ == "__main__":

    run(total_loss_accum, total_tokens_accum)
    # valid()

