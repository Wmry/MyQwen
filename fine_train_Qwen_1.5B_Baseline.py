import logging

from safetensors.torch import save_model
from scipy.ndimage import label
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, PreTrainedModel
import torch
from Model import KGQwen2Attention, KGQwen2DecoderLayer
import torch.optim as optim
from My_Unit import load_config, load_base_model, smart_to_dtype_and_device, load_my_dataset, \
    load_my_dataset_hugging_face_method, print_trainable_parameters
import math
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model, TaskType
import os
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


def run(model, dataloader, tokenizer):
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    model.train()
    for data in dataloader:
        data = prepare_inputs(data, model)
        input_ids = data['input_ids']
        outputs = model(input_ids=input_ids, attention_mask=data['attention_mask'], tokenizer=tokenizer)


def apply_lora(model_tmp: PreTrainedModel):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            # KGQwen2DecoderLayer中的encode_relation相关参数
            "encode_relation.W_q",
            "encode_relation.W_k",
            "encode_relation.W_v",
            "encode_relation.update",

            # lm_head参数
            "lm_head"
        ],
        # 指定需要训练的基础模型层
        modules_to_save=["lm_head"]  # 确保lm_head参数被训练
    )
    model_tmp = get_peft_model(model_tmp, lora_config)
    model_tmp.print_trainable_parameters()
    return model_tmp


# =========================
# 2. 评价指标（PPL）
# =========================
total_loss_accum = 0.0
total_tokens_accum = 0

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

    # 返回空字典，Trainer 不会存储 logits
    return {}


if __name__ == "__main__":
    # =========================
    # 加载模型与数据
    # =========================
    params = load_config("./params.xml")
    tokenizer, model = load_model(params)

    train_path = params['path_set']['train_data']
    train_txtfile = params['path_set']['txtfile_name']
    model_output_path = params['path_set']['output_dir']
    model_train_log = params['path_set']['logging_dir']

    train_dataset, valid_dataset, test_dataset, data_collator = load_my_dataset_hugging_face_method(
        txt_path=train_path,
        txt_name=train_txtfile,
        tokenizer=tokenizer,
        target_multiple=512
    )

    train_length = len(train_dataset)
    train_batch_size = 16

    # =========================
    # 应用 LoRA
    # =========================
    model = apply_lora(model)
    print_trainable_parameters(model)
    # =========================
    # 开启 Gradient Checkpointing
    # =========================
    model.gradient_checkpointing_enable()

    # =========================
    # 训练参数
    # =========================
    training_args = TrainingArguments(
        output_dir="./output",  # 输出目录
        overwrite_output_dir=True,  # 覆盖旧输出
        num_train_epochs=3,  # 训练 epoch
        per_device_train_batch_size=8,  # 训练 batch（可适当调大，看显存）
        per_device_eval_batch_size=1,  # 验证 batch，小一点避免 OOM
        gradient_accumulation_steps=4,  # 累积梯度，相当于扩大 batch

        fp16=False,  # 不用 fp16
        bf16=True,  # 用 bf16（A100/8.9 支持，数值更稳定）
        gradient_checkpointing=True,  # 启用梯度检查点，省显存

        evaluation_strategy="steps",  # 按 step 验证
        eval_steps=256,  # 验证间隔
        save_steps=512,  # 保存间隔（必须是 eval_steps 的倍数）
        load_best_model_at_end=True,  # 保存最优模型
        metric_for_best_model="loss",  # 以 loss 作为最优标准
        greater_is_better=False,

        save_total_limit=2,  # 最多保留 2 个 checkpoint

        logging_dir="./logs",  # 日志
        logging_steps=50,  # 每 50 步记录一次

        # 🔑 避免 eval logits 堆积爆显存
        eval_accumulation_steps=None,  # 每 32 个 batch 把 logits 搬到 CPU
        include_inputs_for_metrics=False,  # 不保存输入到 metrics
        remove_unused_columns=False,  # 减少数据集多余拷贝
        dataloader_num_workers=2,  # 多线程数据加载
        dataloader_pin_memory=True,  # 固定内存加速
    )

    # =========================
    # Trainer
    # =========================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
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

    # 清空累积指标，供下一次验证使用
    total_loss_accum = 0.0
    total_tokens_accum = 0

    # =========================
    # 绘制曲线
    # =========================
    logs = trainer.state.log_history
    epochs, ppl, losses = [], [], []

    for entry in logs:
        if "epoch" in entry:
            if "eval_perplexity" in entry:
                epochs.append(entry["epoch"])
                ppl.append(entry["eval_perplexity"])
            if "loss" in entry:
                losses.append(entry["loss"])

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, ppl, marker="o", label="Eval Perplexity")
    plt.plot(range(len(losses)), losses, linestyle="--", label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.title("Training & Evaluation Curve")
    plt.show()
