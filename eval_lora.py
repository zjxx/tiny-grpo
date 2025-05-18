import json
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
import argparse
import time
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from peft import PeftModel
from tiny import (
    system_prompt,
    compute_score,
    load_model,
)

def load_lora_model(
    base_model_path: str,
    lora_weights_path: str,
    device_map: str = "cuda:0",
) -> tuple[PeftModel, PreTrainedTokenizer]:
    """加载基础模型和LoRA权重"""
    print(f"加载基础模型: {base_model_path}")
    base_model, tokenizer = load_model(
        base_model_path,
        device_map=device_map,
        use_lora=False  # 先加载基础模型
    )
    
    print(f"加载LoRA权重: {lora_weights_path}")
    model = PeftModel.from_pretrained(
        base_model,
        lora_weights_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map
    )
    model.eval()
    
    return model, tokenizer

@torch.no_grad()
def evaluate_lora(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    eval_dataset: pd.DataFrame,
    batch_size: int = 4,
    max_length: int = 1024,
    temperature: float = 0,
    top_p: float = 0.9,
) -> dict:
    """评估LoRA模型"""
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
            
            # tokenize
            inputs = tokenizer(
                [chat_prompt],
                return_tensors="pt",
                padding=True,
                padding_side="left",
                return_attention_mask=True,
            ).to(model.device)
            
            # 生成回答
            generation_config = GenerationConfig(
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                max_length=max_length,
                pad_token_id=tokenizer.eos_token_id,
            )
            
            outputs = model.generate(**inputs, generation_config=generation_config)
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

def main():
    parser = argparse.ArgumentParser(description='评估LoRA模型')
    parser.add_argument('--base_model', type=str, default="/root/autodl-tmp/Qwen2.5-7B", help='基础模型路径')
    parser.add_argument('--lora_weights', type=str, required=True, help='LoRA权重路径')
    parser.add_argument('--eval_data', type=str, default="data/aime_2024_problems.parquet", help='评估数据集路径')
    parser.add_argument('--batch_size', type=int, default=4, help='评估批次大小')
    parser.add_argument('--max_length', type=int, default=1024, help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=0, help='生成温度')
    parser.add_argument('--top_p', type=float, default=0.9, help='top-p采样参数')
    args = parser.parse_args()

    # 创建输出目录
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_dir = Path("./eval_output") / f"eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录已创建: {output_dir}")

    # 加载模型
    device = torch.device("cuda:0")
    model, tokenizer = load_lora_model(
        args.base_model,
        args.lora_weights,
        device_map=device
    )
    
    # 加载评估数据集
    print(f"加载评估数据集: {args.eval_data}")
    eval_dataset = pd.read_parquet(args.eval_data)
    print(f"加载了 {len(eval_dataset)} 个评估样本")
    
    # 进行评估
    eval_metrics = evaluate_lora(
        model,
        tokenizer,
        eval_dataset,
        batch_size=args.batch_size,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    # 保存评估结果
    eval_results_path = output_dir / f"eval_metrics_{timestamp}.json"
    with open(eval_results_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, ensure_ascii=False, indent=2)
    print(f"评估指标已保存到: {eval_results_path}")
    
    # 保存详细结果
    eval_results_df_path = output_dir / f"eval_results_{timestamp}.parquet"
    results_df = pd.DataFrame(eval_metrics["detailed_results"])
    results_df.to_parquet(str(eval_results_df_path))
    print(f"详细评估结果已保存到: {eval_results_df_path}")
    
    # 打印总体评估结果
    print("\n评估完成:")
    print(f"总样本数: {eval_metrics['eval_total_samples']}")
    print(f"平均准确率: {eval_metrics['eval_accuracy']:.4f}")

if __name__ == "__main__":
    main() 