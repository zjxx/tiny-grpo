from collections.abc import Callable
import json
from pathlib import Path
import random
import re
from typing import Any, Iterator, Optional
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import time
from tqdm import tqdm
import argparse
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from loss import approx_kl_divergence, GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch
from torch.cuda.amp import autocast, GradScaler
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np


def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = True,
    bf16: bool = True,
    device_map=None,
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
    )
    
    if use_lora:
        # 配置LoRA
        lora_config = LoraConfig(
            r=lora_r,  # LoRA秩
            lora_alpha=lora_alpha,  # LoRA alpha参数
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 需要应用LoRA的模块
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # 准备模型进行LoRA训练
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        
        # 打印可训练参数信息
        model.print_trainable_parameters()
    
    return model, tokenizer


# DeepSeek Zero system prompt
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""



def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)


def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """计算优势值，处理标准差为0的情况"""
    mean = returns.mean()
    std = returns.std()
    
    # 如果标准差接近0，直接返回中心化的returns
    if std < eps:
        return returns - mean
        
    return (returns - mean) / (std + eps)


def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def sequences_log_probs(
    model: AutoModelForCausalLM,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    output = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = output["logits"]
    log_probs = sequence_log_probs_from_logits(
        logits=logits[:, :-1].to(torch.float32),
        output_ids=sequence_ids[:, 1:],
    )
    return log_probs


def read_jsonl(file_name: str | Path) -> Iterator:
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_prompts(
    file_name: str,
    predicate: Optional[Callable[[Any], bool]] = None,
    max_rows: Optional[int] = None,
) -> list:
    rows = []
    for x in read_jsonl(file_name):
        if predicate is None or predicate(x):
            rows.append(x)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def compute_score(prediction: str, ground_truth: Any) -> dict:

    """计算预测答案与标准答案的匹配分数"""
    # 确保ground_truth是字符串类型
    ground_truth = str(ground_truth).strip()
    
    # 提取<answer>标签中的内容
    pred_match = re.search(r"<answer>(.*?)</answer>", prediction, flags=re.DOTALL)
    pred_answer = pred_match.group(1).strip() if pred_match else prediction.strip()
    print(f"prediction: {pred_answer}")
    print(f"ground_truth: {ground_truth}")
    # 计算分数
    if pred_answer == ground_truth:
        acc = 1.0
    else:
        acc = 0
    
    return {
        "acc": acc,
        "prediction": pred_answer,
        "ground_truth": ground_truth
    }


@torch.no_grad()
def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    eval_dataset: pd.DataFrame,
    batch_size: int = 4,
    max_length: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
) -> dict:
    """评估模型
    
    Args:
        model: 要评估的模型
        tokenizer: 分词器
        eval_dataset: 评估数据集
        batch_size: 批处理大小
        max_length: 输入序列的最大长度
        temperature: 生成温度
        top_p: top-p采样参数
        max_new_tokens: 生成的最大新token数
    """
    model.eval()
    results = []
    total_samples = len(eval_dataset)
    
    print("开始评估...")
    print(f"数据集列名: {eval_dataset.columns.tolist()}")
    
    # 确定问题列名
    if 'prompt' in eval_dataset.columns:
        question_col = 'prompt'
        if isinstance(eval_dataset[question_col].iloc[0], list):
            get_question = lambda x: x[0]['content']
        else:
            get_question = lambda x: x
    elif 'Problem' in eval_dataset.columns:
        question_col = 'Problem'
        get_question = lambda x: x
    else:
        raise ValueError(f"找不到问题列，可用的列名: {eval_dataset.columns.tolist()}")
    
    # 确定答案列名
    if 'reward_model' in eval_dataset.columns:
        answer_col = 'reward_model'
        get_answer = lambda x: x['ground_truth']
    elif 'Answer' in eval_dataset.columns:
        answer_col = 'Answer'
        get_answer = lambda x: x
    else:
        raise ValueError(f"找不到答案列，可用的列名: {eval_dataset.columns.tolist()}")
    
    for i in tqdm(range(0, total_samples, batch_size), desc="评估进度"):
        batch = eval_dataset.iloc[i:i + batch_size]
        questions = [get_question(q) for q in batch[question_col].tolist()]
        answers = [get_answer(a) for a in batch[answer_col].tolist()]
        
        for q, a in zip(questions, answers):
            # 构建完整的对话格式
            chat_messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": "What is 2+2?",
                },
                {
                    "role": "assistant",
                    "content": "<think>Let me solve this step by step.\n1. We need to add 2 and 2\n2. 2 + 2 = 4</think>\n<answer>4</answer>",
                },
                {
                    "role": "user",
                    "content": q,
                }
            ]
            
            # 转换为模型输入格式
            chat_prompt = tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
            
            # tokenize并确保正确设置attention mask
            inputs = tokenizer(
                [chat_prompt],
                return_tensors="pt",
                padding=True,
                padding_side="left",
                return_attention_mask=True,
                max_length=max_length,
                truncation=True
            ).to(model.device)
            
            # 对于generate，使用2D attention mask
            attention_mask = inputs.attention_mask  # 已经是2D [batch_size, seq_len]
            
            # 生成回答
            generation_config = GenerationConfig(
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
            
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=attention_mask,  # 2D attention mask for generate
                generation_config=generation_config
            )
            completion = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # 提取答案标签中的内容
            answer_match = re.search(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL)
            if answer_match:
                # 提取标签中的内容并清理
                pred_answer = answer_match.group(1).strip()
                # 尝试提取纯数字
                numbers = re.findall(r'\d+', pred_answer)
                pred_answer = numbers[0] if numbers else pred_answer
            else:
                pred_answer = completion.strip()
            
            # 使用compute_score进行评分
            score = compute_score(pred_answer, a)
            
            results.append({
                "question": q,
                "prediction": score["prediction"],
                "ground_truth": score["ground_truth"],
                "accuracy": score["acc"],
                "full_completion": completion
            })
            
            if len(results) % 10 == 0:  # 每10个样本打印一次详细信息
                print(f"\n样本 {len(results)}:")
                print(f"问题: {q}")
                print(f"生成结果: {completion}")
                print(f"提取的答案: {score['prediction']}")
                print(f"标准答案: {score['ground_truth']}")
                print(f"准确率: {score['acc']}")
    
    # 计算总体指标
    total_accuracy = sum(r["accuracy"] for r in results) / len(results)
    
    return {
        "eval_accuracy": total_accuracy,
        "eval_total_samples": len(results),
        "detailed_results": results
    }


class MathDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 1024):
        # 读取所有数据
        self.df = pd.read_parquet(data_path)
        # 随机采样1/100的数据
        self.df = self.df.sample(frac=0.00008, random_state=43)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = row['prompt'][0]['content']  # 获取prompt内容
        ground_truth = row['reward_model']['ground_truth']
        
        # 构建完整的对话格式
        chat_messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            }
        ]
        
        # 生成标准答案格式
        answer = f"<think>Let me solve this step by step.</think>\n<answer>{ground_truth}</answer>"
        chat_messages.append({"role": "assistant", "content": answer})
        
        # 转换为模型输入格式
        chat_prompt = self.tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
        
        # tokenize并确保正确设置attention mask
        inputs = self.tokenizer(
            chat_prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True  # 确保返回attention mask
        )
        
        # 创建attention mask和action mask
        attention_mask = inputs["attention_mask"][0]
        input_ids = inputs["input_ids"][0]
        
        # 找到assistant回答开始的位置
        assistant_start = len(self.tokenizer.apply_chat_template(
            chat_messages[:-1], tokenize=False, add_generation_prompt=True
        ).split())
        
        # 创建action mask，只对assistant的回答部分计算loss
        action_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        action_mask[assistant_start:] = True
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "action_mask": action_mask,
            "ground_truth": ground_truth
        }


def plot_metrics(metrics_history, save_dir: Path):
    """绘制训练指标图表
    
    Args:
        metrics_history: 包含训练指标的字典列表
        save_dir: 保存图表的目录
    """
    # 检查是否有数据需要绘制
    if not metrics_history:
        print("警告: 没有训练指标数据可供绘制")
        return
        
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 创建保存目录
    plots_dir = save_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 将指标历史转换为DataFrame
    df = pd.DataFrame(metrics_history)
    
    # 定义要绘制的指标组
    metric_groups = {
        'loss_kl': {
            'metrics': ['train/loss', 'train/kl_divergence'],
            'labels': ['Training Loss', 'KL Divergence'],
            'colors': ['blue', 'red'],
            'title': 'Training Loss and KL Divergence',
            'ylabel': 'Value'
        },
        'reward': {
            'metrics': ['train/reward_mean'],
            'labels': ['Average Reward'],
            'colors': ['green'],
            'title': 'Training Reward (with std)',
            'ylabel': 'Reward',
            'std_metric': 'train/reward_std'
        },
        'grad_norm': {
            'metrics': ['train/lora_grad_norm', 'train/base_grad_norm'],
            'labels': ['LoRA Grad Norm', 'Base Model Grad Norm'],
            'colors': ['purple', 'orange'],
            'title': 'Gradient Norms',
            'ylabel': 'Gradient Norm'
        },
        'learning_rate': {
            'metrics': ['train/learning_rate'],
            'labels': ['Learning Rate'],
            'colors': ['brown'],
            'title': 'Learning Rate Schedule',
            'ylabel': 'Learning Rate',
            'yscale': 'log'
        },
        'log_probs': {
            'metrics': ['train/log_probs_mean', 'train/ref_log_probs_mean'],
            'labels': ['Current Model Log Prob', 'Reference Model Log Prob'],
            'colors': ['blue', 'red'],
            'title': 'Log Probability Statistics',
            'ylabel': 'Log Probability',
            'std_metrics': ['train/log_probs_std', 'train/ref_log_probs_std']
        }
    }
    
    # 绘制每组指标
    for group_name, group_config in metric_groups.items():
        # 检查所需的指标是否都存在
        required_metrics = group_config['metrics']
        if not all(metric in df.columns for metric in required_metrics):
            print(f"Warning: Missing metrics for {group_name}, skipping")
            continue
            
        plt.figure(figsize=(12, 6))
        
        # 绘制主要指标
        for metric, label, color in zip(required_metrics, group_config['labels'], group_config['colors']):
            plt.plot(df[metric], label=label, color=color)
            
            # 如果有标准差指标，绘制置信区间
            if 'std_metric' in group_config and group_config['std_metric'] in df.columns:
                std_metric = group_config['std_metric']
                plt.fill_between(
                    range(len(df)),
                    df[metric] - df[std_metric],
                    df[metric] + df[std_metric],
                    alpha=0.2,
                    color=color
                )
            elif 'std_metrics' in group_config:
                std_metrics = group_config['std_metrics']
                for i, (std_metric, metric) in enumerate(zip(std_metrics, required_metrics)):
                    if std_metric in df.columns:
                        plt.fill_between(
                            range(len(df)),
                            df[metric] - df[std_metric],
                            df[metric] + df[std_metric],
                            alpha=0.2,
                            color=group_config['colors'][i]
                        )
        
        plt.title(group_config['title'])
        plt.xlabel('Batch')
        plt.ylabel(group_config['ylabel'])
        plt.legend()
        plt.grid(True)
        
        # 设置y轴刻度
        if 'yscale' in group_config:
            plt.yscale(group_config['yscale'])
            
        plt.savefig(plots_dir / f'{group_name}_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 绘制优势值分布（如果有数据）
    if 'train/advantage_mean' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df['train/advantage_mean'], bins=50, kde=True)
        plt.title('Advantage Distribution')
        plt.xlabel('Advantage')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(plots_dir / 'advantage_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 绘制评估准确率曲线（如果有评估数据）
    if 'eval/accuracy' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df['eval/accuracy'], label='Evaluation Accuracy', color='green', marker='o')
        plt.title('Evaluation Accuracy')
        plt.xlabel('Evaluation Point')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / 'eval_accuracy_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"All plots have been saved to: {plots_dir}")


def main():
    # 设置PyTorch显存分配策略
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    gc.collect()
    
    parser = argparse.ArgumentParser(description='训练或评估数学问题求解模型')
    parser.add_argument('--eval_only', action='store_true', help='只进行评估，不进行训练')
    parser.add_argument('--model_path', type=str, default=None, help='评估时使用的模型路径，如果不指定则使用默认模型')
    parser.add_argument('--use_lora', action='store_true', help='是否使用LoRA进行训练')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA的秩')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA的alpha参数')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA的dropout率')
    parser.add_argument('--temperature', type=float, default=0.7, help='生成温度，必须大于0')
    parser.add_argument('--use_amp', action='store_true', help='是否使用混合精度训练')
    parser.add_argument('--offload_to_cpu', action='store_true', help='是否将部分张量卸载到CPU')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--lr', type=float, default=1e-6, help='学习率')
    parser.add_argument('--lora_grad_clip', type=float, default=0.1, help='LoRA参数的梯度裁剪阈值')
    parser.add_argument('--eval_max_length', type=int, default=2048, help='评估时的最大序列长度')
    parser.add_argument('--eval_max_new_tokens', type=int, default=1024, help='评估时生成的最大新token数')
    parser.add_argument('--eval_temperature', type=float, default=0.7, help='评估时的生成温度')
    parser.add_argument('--eval_top_p', type=float, default=0.9, help='评估时的top-p采样参数')
    parser.add_argument('--use_wandb', action='store_true', help='是否使用wandb进行实验跟踪')
    parser.add_argument('--wandb_project', type=str, default='tiny_grpo', help='wandb项目名称')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb实体名称(用户名或团队名)')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb运行名称')
    args = parser.parse_args()

    seed = 42
    device_index = 0
    model_name = args.model_path if args.model_path else "/root/autodl-tmp/Qwen2.5-7B"
    
    # 创建带时间戳的输出目录
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    base_output_dir = Path("./output")
    checkpoint_path = base_output_dir / f"checkpoint_{timestamp}"
    
    # 确保输出目录存在
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    print(f"输出目录已创建: {checkpoint_path}")

    # 初始化wandb
    if args.use_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name or f"run_{timestamp}",
                config={
                    "model_name": model_name,
                    "use_lora": args.use_lora,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "learning_rate": args.lr,
                    "batch_size": train_batch_size,
                    "max_length": max_length,
                    "eval_max_length": args.eval_max_length,
                    "eval_max_new_tokens": args.eval_max_new_tokens,
                    "eval_temperature": args.eval_temperature,
                    "eval_top_p": args.eval_top_p,
                    "grad_clip": args.grad_clip,
                    "lora_grad_clip": args.lora_grad_clip,
                    "seed": seed,
                }
            )
            print("成功初始化wandb")
        except Exception as e:
            print(f"wandb初始化失败: {e}")
            print("继续运行但不使用wandb")
            args.use_wandb = False
    else:
        print("未启用wandb,跳过初始化")

    checkpoint_interval = 20
    train_batch_size = 4
    lr = args.lr  # 使用命令行参数中的学习率
    kl_weight = 0.01
    clip_eps = 0.2
    max_length = 512
    num_epochs = 5
    grad_clip = args.grad_clip  # 使用命令行参数中的梯度裁剪阈值

    device = torch.device("cuda", device_index)
    cpu_device = torch.device("cpu")
    init_rng(seed)

    # 初始化混合精度训练
    scaler = GradScaler() if args.use_amp else None
    
    # 加载模型，添加LoRA相关参数
    model, tokenizer = load_model(
        model_name, 
        device_map=device,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    # 在训练循环开始前初始化指标历史记录
    metrics_history = []
    
    if not args.eval_only:
        # 训练模式
        reference_model, _ = load_model(
            model_name, 
            device_map="cpu" if args.offload_to_cpu else device,
            use_lora=False  # 参考模型不使用LoRA
        )
        # old_model用于采样
        old_model, _ = load_model(
            model_name, 
            device_map="cpu" if args.offload_to_cpu else device,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
        old_model.load_state_dict(model.state_dict())
        old_model.eval()
        
        # 启用更激进的梯度检查点
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
                "checkpoint_every_layer": True,
                "preserve_rng_state": False,
                "deterministic": True
            }
        )
        
        # 分离LoRA参数和基础模型参数
        lora_params = []
        base_params = []
        for name, param in model.named_parameters():
            if 'lora_' in name:
                lora_params.append(param)
            else:
                base_params.append(param)
        
        # 为LoRA参数和基础模型参数使用不同的优化器设置
        optimizer = optim.AdamW([
            {'params': lora_params, 'lr': lr, 'weight_decay': 0.01, 'betas': (0.9, 0.999)},
            {'params': base_params, 'lr': lr * 0.1, 'weight_decay': 0.0, 'betas': (0.9, 0.999)}
        ], eps=1e-8)
        
        # 添加学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=100,
            T_mult=2,
            eta_min=lr * 0.1
        )
        
        reference_model.eval()

        # 加载训练数据集
        train_dataset = MathDataset("data/dapo-math-17k.parquet", tokenizer, max_length)
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )
        
        print(f"加载了 {len(train_dataset)} 个训练样本")
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            total_kl = 0
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                # 每个batch开始时清理缓存
                if batch_idx % 10 == 0:  # 每10个batch清理一次
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # 优化attention mask处理
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                action_mask = batch["action_mask"].to(device)
                
                # 1. 用 old_model 采样，获得采样序列和 action_log_probs
                rewards = []
                sampled_action_log_probs = []
                max_attempts = 10  # 最大尝试次数
                attempt = 0
                while attempt < max_attempts:
                    batch_rewards = []
                    batch_log_probs = []
                    for input_id, gt in zip(batch["input_ids"], batch["ground_truth"]):
                        # 优化attention mask处理，避免重复创建
                        attention_mask_2d = (input_id != tokenizer.pad_token_id).to(device)
                        attention_mask_2d = attention_mask_2d.unsqueeze(0)  # [1, seq_len]
                        
                        # 使用新的autocast API
                        with torch.amp.autocast('cuda', enabled=args.use_amp):
                            # 如果启用CPU卸载，将输入移到CPU
                            if args.offload_to_cpu:
                                input_id = input_id.cpu()
                                attention_mask_2d = attention_mask_2d.cpu()
                            
                            outputs = old_model.generate(
                                input_ids=input_id.unsqueeze(0).to(device if not args.offload_to_cpu else "cpu"),
                                attention_mask=attention_mask_2d,
                                max_new_tokens=128,
                                do_sample=True,
                                top_p=0.9,
                                temperature=0.7,
                                pad_token_id=tokenizer.eos_token_id
                            )
                            
                            # 生成完成后立即将结果移回GPU
                            outputs = outputs.to(device)
                        
                        completion = tokenizer.decode(outputs[0][input_id.shape[0]:], skip_special_tokens=True)
                        gt_str = str(gt).strip()
                        pred_match = re.search(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL)
                        pred_answer = pred_match.group(1).strip() if pred_match else completion.strip()
                        reward = 1.0 if pred_answer == gt_str else -1.0
                        batch_rewards.append(reward)
                        
                        # 计算 old_model 下的 log_probs，使用混合精度
                        with torch.no_grad(), torch.amp.autocast('cuda', enabled=args.use_amp):
                            # 如果启用CPU卸载，将输入移到CPU
                            if args.offload_to_cpu:
                                input_id = input_id.cpu()
                                attention_mask_2d = attention_mask_2d.cpu()
                            
                            out = old_model(
                                input_ids=input_id.unsqueeze(0).to(device if not args.offload_to_cpu else "cpu"),
                                attention_mask=attention_mask_2d,
                                use_cache=False,
                            )
                            logits = out.logits.to(device)  # 将logits移回GPU
                            log_probs = sequence_log_probs_from_logits(
                                logits=logits[:, :-1].to(torch.float32),
                                output_ids=input_id[1:].unsqueeze(0).to(device)
                            )
                            batch_log_probs.append(log_probs.squeeze(0))
                            
                            # 清理不需要的张量
                            del out, logits
                            torch.cuda.empty_cache()
                    
                    # 检查是否有不同的reward值
                    if len(set(batch_rewards)) > 1:
                        rewards = batch_rewards
                        sampled_action_log_probs = batch_log_probs
                        print(f"成功收集到不同的rewards: {rewards}")
                        break
                    else:
                        print(f"尝试 {attempt + 1}/{max_attempts}: 所有rewards值相同 ({batch_rewards[0]})，重新采样")
                        attempt += 1
                
                if attempt == max_attempts:
                    print("达到最大尝试次数，跳过当前batch")
                    continue
                
                rewards = torch.tensor(rewards, device=device)
                advantages = group_advantages(rewards)
                sampled_action_log_probs = torch.stack(sampled_action_log_probs)
                optimizer.zero_grad(set_to_none=True)  # 使用set_to_none=True来更彻底地清理梯度
                
                # 2. 用当前 model 前向传播，计算 log_probs
                # 优化attention mask处理，避免重复创建
                attention_mask_4d = attention_mask.unsqueeze(1).unsqueeze(2).to(model.dtype)
                
                # 使用新的autocast API
                with torch.amp.autocast('cuda', enabled=args.use_amp):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask_4d,
                        use_cache=False,
                    )
                    logits = outputs.logits
                    
                    # 检查logits是否包含NaN
                    if torch.isnan(logits).any():
                        print(f"警告: logits包含NaN值，跳过当前batch")
                        continue
                        
                    log_probs = sequence_log_probs_from_logits(
                        logits=logits[:, :-1].to(torch.float32),
                        output_ids=input_ids[:, 1:]
                    )
                    
                    # 检查log_probs是否包含NaN
                    if torch.isnan(log_probs).any():
                        print(f"警告: log_probs包含NaN值，跳过当前batch")
                        continue
                    
                    # 3. 参考模型log_probs
                    with torch.no_grad():
                        # 如果启用CPU卸载，将输入移到CPU
                        if args.offload_to_cpu:
                            input_ids = input_ids.cpu()
                            attention_mask_4d = attention_mask_4d.cpu()
                        
                        ref_outputs = reference_model(
                            input_ids=input_ids.to(device if not args.offload_to_cpu else "cpu"),
                            attention_mask=attention_mask_4d.to(reference_model.dtype),
                            use_cache=False,
                        )
                        ref_logits = ref_outputs.logits.to(device)  # 将logits移回GPU
                        
                        # 检查ref_logits是否包含NaN
                        if torch.isnan(ref_logits).any():
                            print(f"警告: ref_logits包含NaN值，跳过当前batch")
                            continue
                            
                        ref_log_probs = sequence_log_probs_from_logits(
                            logits=ref_logits[:, :-1].to(torch.float32),
                            output_ids=input_ids[:, 1:].to(device)
                        )
                        
                        # 检查ref_log_probs是否包含NaN
                        if torch.isnan(ref_log_probs).any():
                            print(f"警告: ref_log_probs包含NaN值，跳过当前batch")
                            continue
                        
                        # 清理不需要的张量
                        del ref_outputs, ref_logits
                        torch.cuda.empty_cache()
                    
                    # 4. 构造Experience
                    experience = Experience(
                        sequences=input_ids.to(device),
                        action_log_probs=sampled_action_log_probs.detach(),
                        log_probs_ref=ref_log_probs,
                        returns=rewards,
                        advantages=advantages,
                        attention_mask=attention_mask[:, 1:],
                        action_mask=action_mask[:, 1:]
                    )
                    
                    # 检查experience中的值是否包含NaN
                    for key, value in experience.__dict__.items():
                        if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                            print(f"警告: experience.{key}包含NaN值，跳过当前batch")
                            continue
                    
                    grpo_loss_fn = GRPOLoss(clip_eps, kl_weight)
                    loss, kl = grpo_loss_fn(log_probs, experience)
                
                # 使用混合精度进行反向传播
                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    
                    # 分别处理LoRA参数和基础模型参数的梯度
                    lora_grad_norm = 0.0
                    base_grad_norm = 0.0
                    
                    # 计算LoRA参数的梯度范数
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if 'lora_' in name:
                                # 对LoRA参数使用更严格的梯度裁剪
                                if torch.isnan(param.grad).any():
                                    print(f"警告: LoRA参数 {name} 的梯度包含NaN值，跳过当前batch")
                                    optimizer.zero_grad(set_to_none=True)
                                    continue
                                lora_grad_norm += param.grad.norm(2).item() ** 2
                            else:
                                base_grad_norm += param.grad.norm(2).item() ** 2
                    
                    lora_grad_norm = lora_grad_norm ** 0.5
                    base_grad_norm = base_grad_norm ** 0.5
                    
                    # 分别处理LoRA参数和基础模型参数的梯度裁剪
                    if lora_grad_norm > args.lora_grad_clip:
                        scale = args.lora_grad_clip / (lora_grad_norm + 1e-6)
                        for name, param in model.named_parameters():
                            if param.grad is not None and 'lora_' in name:
                                param.grad.mul_(scale)
                    
                    if base_grad_norm > grad_clip:
                        scale = grad_clip / (base_grad_norm + 1e-6)
                        for name, param in model.named_parameters():
                            if param.grad is not None and 'lora_' not in name:
                                param.grad.mul_(scale)
                    
                    # 检查最终梯度是否包含NaN
                    has_nan = False
                    for name, param in model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            print(f"警告: 参数 {name} 的梯度包含NaN值，跳过当前batch")
                            has_nan = True
                            break
                    
                    if has_nan:
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    
                    # 分别处理LoRA参数和基础模型参数的梯度
                    lora_grad_norm = 0.0
                    base_grad_norm = 0.0
                    
                    # 计算LoRA参数的梯度范数
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if 'lora_' in name:
                                if torch.isnan(param.grad).any():
                                    print(f"警告: LoRA参数 {name} 的梯度包含NaN值，跳过当前batch")
                                    optimizer.zero_grad(set_to_none=True)
                                    continue
                                lora_grad_norm += param.grad.norm(2).item() ** 2
                            else:
                                base_grad_norm += param.grad.norm(2).item() ** 2
                    
                    lora_grad_norm = lora_grad_norm ** 0.5
                    base_grad_norm = base_grad_norm ** 0.5
                    
                    # 分别处理LoRA参数和基础模型参数的梯度裁剪
                    if lora_grad_norm > args.lora_grad_clip:
                        scale = args.lora_grad_clip / (lora_grad_norm + 1e-6)
                        for name, param in model.named_parameters():
                            if param.grad is not None and 'lora_' in name:
                                param.grad.mul_(scale)
                    
                    if base_grad_norm > grad_clip:
                        scale = grad_clip / (base_grad_norm + 1e-6)
                        for name, param in model.named_parameters():
                            if param.grad is not None and 'lora_' not in name:
                                param.grad.mul_(scale)
                    
                    # 检查最终梯度是否包含NaN
                    has_nan = False
                    for name, param in model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            print(f"警告: 参数 {name} 的梯度包含NaN值，跳过当前batch")
                            has_nan = True
                            break
                    
                    if has_nan:
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    
                    optimizer.step()
                
                # 更新学习率
                scheduler.step()
                
                # 清理不需要的张量
                del outputs, logits, log_probs, ref_log_probs, experience
                torch.cuda.empty_cache()
                
                total_loss += loss.item()
                total_kl += kl.item()
                if (batch_idx + 1) % 2 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    avg_kl = total_kl / (batch_idx + 1)
                    current_lr = scheduler.get_last_lr()[0]
                    
                    # 计算当前batch的平均reward
                    avg_reward = rewards.mean().item()
                    reward_std = rewards.std().item()
                    
                    # 准备要记录的指标
                    current_metrics = {
                        "train/loss": avg_loss,
                        "train/kl_divergence": avg_kl,
                        "train/reward_mean": avg_reward,
                        "train/reward_std": reward_std,
                        "train/lora_grad_norm": lora_grad_norm,
                        "train/base_grad_norm": base_grad_norm,
                        "train/learning_rate": current_lr,
                        "train/epoch": epoch + 1,
                        "train/batch": batch_idx + 1,
                    }
                    
                    # 安全地添加其他指标
                    try:
                        current_metrics.update({
                            "train/advantage_mean": advantages.mean().item(),
                            "train/advantage_std": advantages.std().item(),
                        })
                    except (AttributeError, RuntimeError) as e:
                        print(f"警告: 无法记录优势值指标: {e}")
                    
                    try:
                        if 'log_probs' in locals() and log_probs is not None:
                            current_metrics.update({
                                "train/log_probs_mean": log_probs.mean().item(),
                                "train/log_probs_std": log_probs.std().item(),
                            })
                    except (AttributeError, RuntimeError) as e:
                        print(f"警告: 无法记录log_probs指标: {e}")
                    
                    try:
                        if 'ref_log_probs' in locals() and ref_log_probs is not None:
                            current_metrics.update({
                                "train/ref_log_probs_mean": ref_log_probs.mean().item(),
                                "train/ref_log_probs_std": ref_log_probs.std().item(),
                            })
                    except (AttributeError, RuntimeError) as e:
                        print(f"警告: 无法记录ref_log_probs指标: {e}")
                    
                    print(f"Epoch {epoch+1}, Batch {batch_idx+1}, "
                          f"Loss: {avg_loss:.4f}, KL: {avg_kl:.4f}, "
                          f"Reward: {avg_reward:.4f} ± {reward_std:.4f}, "
                          f"LoRA Grad Norm: {lora_grad_norm:.4f}, Base Grad Norm: {base_grad_norm:.4f}, "
                          f"LR: {current_lr:.2e}")
                    
                    # 将当前指标添加到历史记录
                    metrics_history.append(current_metrics)
                    
                    # 如果启用了wandb,记录指标
                    if args.use_wandb:
                        wandb.log(current_metrics)
            # 5. 每个epoch结束后，同步old_model参数
            old_model.load_state_dict(model.state_dict())
            # 训练结束后保存模型和分词器
            if args.use_lora:
                # 保存LoRA权重和配置
                lora_save_path = checkpoint_path / 'lora_weights'
                lora_save_path.mkdir(parents=True, exist_ok=True)
                
                # 保存模型权重和配置
                model.save_pretrained(
                    str(lora_save_path),
                    safe_serialization=True,  # 使用safetensors格式
                    max_shard_size="2GB"  # 分片大小
                )
                
                # 保存tokenizer
                tokenizer_save_path = checkpoint_path / 'tokenizer'
                tokenizer_save_path.mkdir(parents=True, exist_ok=True)
                tokenizer.save_pretrained(str(tokenizer_save_path))
                
                print(f"LoRA权重和配置已保存到: {lora_save_path}")
                print(f"Tokenizer已保存到: {tokenizer_save_path}")
                
                # 验证保存的文件
                required_files = ['adapter_config.json', 'adapter_model.safetensors']
                for file in required_files:
                    if not (lora_save_path / file).exists():
                        print(f"警告: 未找到必要的文件 {file}")
                    else:
                        print(f"已确认文件存在: {file}")
            else:
                # 保存完整模型
                model_save_path = checkpoint_path / 'full_model'
                model_save_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(
                    str(model_save_path),
                    safe_serialization=True,
                    max_shard_size="2GB"
                )
                
                # 保存tokenizer
                tokenizer_save_path = checkpoint_path / 'tokenizer'
                tokenizer_save_path.mkdir(parents=True, exist_ok=True)
                tokenizer.save_pretrained(str(tokenizer_save_path))
                
                print(f"完整模型已保存到: {model_save_path}")
                print(f"Tokenizer已保存到: {tokenizer_save_path}")
    else:
        print("评估模式：跳过训练，直接进行评估")

    
     # 保存训练指标历史
    metrics_path = checkpoint_path / f"metrics_history_{timestamp}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_history, f, ensure_ascii=False, indent=2)
    print(f"训练指标历史已保存到: {metrics_path}")
    
    # 同时保存为parquet格式，方便后续分析
    if metrics_history:
        metrics_df = pd.DataFrame(metrics_history)
        metrics_parquet_path = checkpoint_path / f"metrics_history_{timestamp}.parquet"
        metrics_df.to_parquet(str(metrics_parquet_path))
        print(f"训练指标历史(parquet格式)已保存到: {metrics_parquet_path}")
    
    # 训练和评估完成后绘制图表
    plot_metrics(metrics_history, checkpoint_path)
    # 评估部分
    print("\n开始评估...")
    eval_dataset = pd.read_parquet("data/aime_2024_problems.parquet")
    print(f"加载了 {len(eval_dataset)} 个评估样本")
    
    eval_metrics = evaluate(
        model,
        tokenizer,
        eval_dataset,
        batch_size=train_batch_size,
        max_length=args.eval_max_length,
        temperature=args.eval_temperature,
        top_p=args.eval_top_p,
        max_new_tokens=args.eval_max_new_tokens,
    )
    
    # 如果启用了wandb,记录评估指标
    if args.use_wandb:
        wandb.log({
            "eval/accuracy": eval_metrics["eval_accuracy"],
            "eval/total_samples": eval_metrics["eval_total_samples"],
            "eval/epoch": num_epochs,
        })
        
        # 记录每个样本的详细评估结果
        for i, result in enumerate(eval_metrics["detailed_results"]):
            wandb.log({
                f"eval_samples/sample_{i}/accuracy": result["accuracy"],
                f"eval_samples/sample_{i}/question": result["question"],
                f"eval_samples/sample_{i}/prediction": result["prediction"],
                f"eval_samples/sample_{i}/ground_truth": result["ground_truth"],
            })
    
    # 如果启用了wandb,上传图表
    if args.use_wandb:
        plots_dir = checkpoint_path / 'plots'
        for plot_file in plots_dir.glob('*.png'):
            wandb.log({f"plots/{plot_file.stem}": wandb.Image(str(plot_file))})
    
   
    
    # 保存评估指标
    eval_results_path = checkpoint_path / f"eval_metrics_{timestamp}.json"
    with open(eval_results_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, ensure_ascii=False, indent=2)
    print(f"评估指标已保存到: {eval_results_path}")
    
    # 保存评估详细结果
    eval_results_df_path = checkpoint_path / f"eval_results_{timestamp}.parquet"
    results_df = pd.DataFrame(eval_metrics.get("detailed_results", []))
    if not results_df.empty:
        results_df.to_parquet(str(eval_results_df_path))
        print(f"详细评估结果已保存到: {eval_results_df_path}")


if __name__ == "__main__":
    main()
