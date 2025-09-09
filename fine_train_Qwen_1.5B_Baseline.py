from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
from Model import KGQwen2Attention
import torch.optim as optim
from My_Unit import load_config, load_base_model, smart_to_dtype_and_device, load_my_dataset
import math
import matplotlib.pyplot as plt

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
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    model.train()
    for data in dataloader:
        data = prepare_inputs(data, model)
        input_ids = data['input_ids']
        outputs = model(input_ids=input_ids, attention_mask=data['attention_mask'],tokenizer=tokenizer)

def setup_selective_training(model_kg_qwen : torch.nn.Module):
    for param in model_kg_qwen.parameters():
        param.requires_grad = False

    for name, module in model_kg_qwen.model.layers.named_modules():
        if isinstance(module, KGQwen2Attention):
            for param_name, param in module.named_parameters():
                if param_name.find('encode_relation') >= 0:
                    print(f"将训练KGQwen2Attention层: {param_name}")
                    param.requires_grad = True

    # 训练 lm_head
    for param in model_kg_qwen.lm_head.parameters():
        param.requires_grad = True

# =========================
# 3. 定义评价指标：困惑度 Perplexity
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = labels[..., 1:].reshape(-1)

    # 忽略 -100 的 label（transformers 默认 mask）
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(
        torch.tensor(shift_logits, dtype=torch.float32),
        torch.tensor(shift_labels, dtype=torch.long),
    )
    perplexity = math.exp(loss.item())
    return {"perplexity": perplexity}

if __name__ == "__main__":
    params = load_config("./params.xml")
    tokenizer, model = load_model(params)
    train_path = params['path_set']['train_data']
    train_txtfile = params['path_set']['txtfile_name']
    train_dataset, test_dataset = load_my_dataset(txt_path=train_path, txt_name=train_txtfile, tokenizer=tokenizer)

    setup_selective_training(model)


    # =========================
    # 4. 设置训练参数
    # =========================
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",  # 每个 epoch 评估
        save_strategy="epoch",  # 每个 epoch 保存
        save_total_limit=2,  # 最多保留2个 checkpoint
        load_best_model_at_end=True,  # 自动加载最佳模型
        metric_for_best_model="perplexity",  # 按 perplexity 选最优
        greater_is_better=False,  # PPL 越小越好
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_strategy="epoch",
        report_to="none",  # 禁止 wandb 报错
    )

    # =========================
    # 5. 定义 Trainer
    # =========================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # =========================
    # 6. 训练并保存最佳模型
    # =========================
    train_result = trainer.train()
    trainer.save_model("./best_model")

    # =========================
    # 7. 绘制指标曲线
    # =========================
    logs = trainer.state.log_history

    epochs = []
    ppl = []
    losses = []

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
