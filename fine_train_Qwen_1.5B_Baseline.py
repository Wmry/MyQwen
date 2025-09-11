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
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = labels[..., 1:].reshape(-1)

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(
        torch.tensor(shift_logits, dtype=torch.float32),
        torch.tensor(shift_labels, dtype=torch.long),
    )
    perplexity = math.exp(loss.item())
    return {"perplexity": perplexity}


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
        tokenizer=tokenizer
    )

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
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="perplexity",
        greater_is_better=False,
        per_device_train_batch_size=5,
        per_device_eval_batch_size=5,
        num_train_epochs=3,
        logging_dir=model_train_log,
        logging_strategy="epoch",
        report_to="none",
        gradient_checkpointing=True,  # 显存大幅下降
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
