import os
import re
import json
import random
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOTrainer, GRPOConfig

# 固定随机种子，保证每次运行训练/测试集一致，实验可复现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pretrain_models", "Qwen2-0.5B-Instruct")
OUTPUT_DIR = os.path.join(BASE_DIR, "gpro_output")
RESULTS_FILE = os.path.join(BASE_DIR, "accuracy_comparison.json")


@dataclass
class MathProblem:
    a: int
    b: int
    op: str
    answer: float
    question: str


def generate_math_problems(n_problems: int = 200, min_num: int = 1, max_num: int = 999,
                          balance_ops: bool = True, multiply_curriculum: bool = False,
                          multiply_weight: int = 1) -> List[MathProblem]:
    problems = []

    if balance_ops:
        base_count = n_problems // (3 + multiply_weight)
        remainder = n_problems - base_count * (3 + multiply_weight)
        op_counts = {'+': base_count, '-': base_count, '*': base_count * multiply_weight, '/': base_count}
        for i in range(remainder):
            op_counts[['+', '-', '*', '/'][i % 4]] += 1
        op_counts['*'] += max(0, n_problems - sum(op_counts.values()))
    else:
        op_counts = {'+': n_problems // 4, '-': n_problems // 4,
                     '*': n_problems // 4, '/': n_problems - 3 * (n_problems // 4)}

    low = max(1, min_num)
    high = max(low, max_num)

    for op, count in op_counts.items():
        for _ in range(count):
            if op == '+':
                a = random.randint(low, high)
                b = random.randint(low, high)
                answer = a + b
            elif op == '-':
                a = random.randint(low, high)
                b = random.randint(low, high)
                a, b = max(a, b), min(a, b)
                answer = a - b
            elif op == '*':
                if multiply_curriculum:
                    difficulty = random.random()
                    if difficulty < 0.35:
                        a = random.randint(2, 9)
                        b = random.randint(2, 9)
                    elif difficulty < 0.6:
                        a = random.randint(2, 9)
                        b = random.randint(2, 19)
                    elif difficulty < 0.82:
                        a = random.randint(2, 9)
                        b = random.randint(10, 99)
                    else:
                        a = random.randint(10, 99)
                        b = random.randint(10, 99)
                else:
                    a = random.randint(low, high)
                    b = random.randint(low, high)
                answer = a * b
            elif op == '/':
                if multiply_curriculum:
                    difficulty = random.random()
                    if difficulty < 0.35:
                        b = random.randint(2, 9)
                        k = random.randint(2, 9)
                    elif difficulty < 0.7:
                        b = random.randint(2, 9)
                        k = random.randint(2, 19)
                    else:
                        b = random.randint(2, 29)
                        k = random.randint(2, 30)
                    a = b * k
                else:
                    b = random.randint(2, max(9, min(high, 99)))
                    k = random.randint(2, max(9, min(high, 50)))
                    a = b * k
                answer = a // b

            question = f"请计算: {a} {op} {b} = ?"
            problems.append(MathProblem(a=a, b=b, op=op, answer=answer, question=question))

    random.shuffle(problems)
    return problems


def extract_number_from_text(text: str) -> float:
    text = text.replace(',', '')
    matches = re.findall(r'-?\d+\.?\d*', text)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None


def is_answer_correct(predicted: float, expected: float, tolerance: float = 1e-2) -> bool:
    if predicted is None:
        return False
    return abs(predicted - expected) <= tolerance


def test_model_accuracy(model, tokenizer, problems: List[MathProblem], device: str = "cuda", batch_size: int = 8) -> Dict:
    model.eval()
    correct = 0
    results = []
    op_stats = {'+': {'correct': 0, 'total': 0},
                '-': {'correct': 0, 'total': 0},
                '*': {'correct': 0, 'total': 0},
                '/': {'correct': 0, 'total': 0}}

    with torch.no_grad():
        for i in tqdm(range(0, len(problems), batch_size), desc="Testing"):
            batch = problems[i:i + batch_size]
            prompts = []

            for p in batch:
                messages = [
                    {"role": "system", "content": "你是一个计算助手，只能输出最终的数字答案，不要输出任何额外的解释或文字。"},
                    {"role": "user", "content": p.question}
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompts.append(text)

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                temperature=None,
                top_p=None,
                eos_token_id=[151645, 151643],
                pad_token_id=151643,
            )

            for j, p in enumerate(batch):
                generated_ids = outputs[j][inputs['input_ids'].shape[1]:]
                answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                predicted_num = extract_number_from_text(answer_text)
                is_correct = is_answer_correct(predicted_num, p.answer)

                if is_correct:
                    correct += 1
                    op_stats[p.op]['correct'] += 1
                op_stats[p.op]['total'] += 1

                results.append({
                    'question': p.question,
                    'expected': p.answer,
                    'predicted_text': answer_text,
                    'predicted_num': predicted_num,
                    'correct': is_correct,
                    'op': p.op
                })

    accuracy = correct / len(problems) if problems else 0

    op_accuracies = {}
    for op, stats in op_stats.items():
        op_accuracies[op] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': len(problems),
        'op_accuracies': op_accuracies,
        'op_stats': op_stats,
        'results': results
    }


def build_gpro_dataset(problems: List[MathProblem]) -> Dataset:
    data = []
    for p in problems:
        messages = [
            {"role": "system", "content": "你是一个计算助手，只输出最终的数字答案，不要输出任何额外的文字、算式或解释。\n"
                                          "示例1：问题「12 + 34 = ?」→ 输出「46」\n"
                                          "示例2：问题「25 × 8 = ?」→ 输出「200」\n"
                                          "示例3：问题「100 ÷ 4 = ?」→ 输出「25」\n"
                                          "记住：只输出数字本身，不要加句号、逗号、单位或其他任何字符。"},
            {"role": "user", "content": p.question}
        ]
        data.append({
            'prompt': messages,
            'expected_answer': p.answer,
            'op': p.op,
        })
    return Dataset.from_list(data)


def reward_fn(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
    expected_answers = kwargs.get("expected_answer", None)
    ops = kwargs.get("op", None)
    num_generations = len(completions) // len(prompts) if len(prompts) > 0 else 1

    if expected_answers is None or len(expected_answers) == 0:
        return [-1.0] * len(completions)

    expanded_expected = []
    expanded_ops = []
    for i, ans in enumerate(expected_answers):
        op = ops[i] if ops is not None and i < len(ops) else '+'
        for _ in range(num_generations):
            expanded_expected.append(float(ans))
            expanded_ops.append(op)

    while len(expanded_expected) < len(completions):
        expanded_expected.append(expanded_expected[-1] if expanded_expected else 0.0)
        expanded_ops.append(expanded_ops[-1] if expanded_ops else '+')
    expanded_expected = expanded_expected[:len(completions)]
    expanded_ops = expanded_ops[:len(completions)]

    rewards = []
    for comp, expected, op in zip(completions, expanded_expected, expanded_ops):
        pred_num = extract_number_from_text(comp)
        if pred_num is None:
            rewards.append(-2.0)
            continue

        abs_error = abs(pred_num - expected)

        if expected != 0:
            rel_error = abs_error / (abs(expected) + 1e-8)
        else:
            rel_error = abs_error

        if abs_error <= 1e-2:
            if op == '*':
                rewards.append(6.0)
            elif op == '/':
                rewards.append(3.5)
            else:
                rewards.append(2.5)
        elif abs_error <= max(0.5, expected * 0.001):
            if op == '*':
                rewards.append(3.5)
            else:
                rewards.append(1.5)
        elif op == '*' and expected >= 1000 and rel_error <= 0.02:
            rewards.append(2.0)
        elif abs_error <= max(1.0, expected * 0.02):
            rewards.append(0.5)
        elif rel_error <= 0.05:
            rewards.append(-0.1)
        elif rel_error <= 0.2:
            rewards.append(-0.8)
        elif rel_error <= 0.5:
            rewards.append(-1.8)
        else:
            rewards.append(-3.0)
    return rewards


def run_gpro_training(model, tokenizer, train_dataset: Dataset):
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def format_sample(sample):
        sample["prompt"] = tokenizer.apply_chat_template(
            sample["prompt"], tokenize=False, add_generation_prompt=True
        )
        return sample

    train_dataset = train_dataset.map(format_sample)

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name="qwen2-0.5b-gpro-math",
        learning_rate=1.5e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=4,
        max_completion_length=32,
        beta=0.04,
        num_generations=4,
        logging_steps=5,
        save_steps=9999,
        save_total_limit=1,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        report_to="none",
        remove_unused_columns=False,
        warmup_steps=5,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"GPRO训练完成，模型已保存至: {OUTPUT_DIR}")

    return trainer.model


def print_comparison_report(baseline_results: Dict, gpro_results: Dict):
    print("\n" + "=" * 70)
    print("                    准确率对比报告")
    print("=" * 70)

    print(f"\n总体准确率:")
    print(f"  GPRO之前:  {baseline_results['accuracy']:.2%}  ({baseline_results['correct']}/{baseline_results['total']})")
    print(f"  GPRO之后:  {gpro_results['accuracy']:.2%}  ({gpro_results['correct']}/{gpro_results['total']})")

    diff = gpro_results['accuracy'] - baseline_results['accuracy']
    diff_pct = (diff / baseline_results['accuracy'] * 100) if baseline_results['accuracy'] > 0 else 0
    arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
    print(f"  变化:      {arrow} {diff:+.2%}  (相对变化 {diff_pct:+.2f}%)")

    print(f"\n各运算符准确率对比:")
    print(f"  {'运算符':<8} {'GPRO之前':<12} {'GPRO之后':<12} {'变化':<15}")
    print(f"  {'-'*45}")

    for op in ['+', '-', '*', '/']:
        base_acc = baseline_results['op_accuracies'].get(op, 0)
        gpro_acc = gpro_results['op_accuracies'].get(op, 0)
        op_diff = gpro_acc - base_acc
        arrow = "↑" if op_diff > 0 else ("↓" if op_diff < 0 else "→")
        print(f"  {op:<8} {base_acc:<12.2%} {gpro_acc:<12.2%} {arrow} {op_diff:+.2%}")

    print(f"\n各运算符详细统计:")
    for op in ['+', '-', '*', '/']:
        base_stats = baseline_results['op_stats'].get(op, {'correct': 0, 'total': 0})
        gpro_stats = gpro_results['op_stats'].get(op, {'correct': 0, 'total': 0})
        print(f"  {op}: 之前 {base_stats['correct']}/{base_stats['total']}  |  之后 {gpro_stats['correct']}/{gpro_stats['total']}")

    print("=" * 70 + "\n")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n加载模型: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)
    print("模型加载完成。")

    print("\n生成数学测试题（含一位数~三位数，训练集乘法×2权重+从乘法口诀表开始课程学习）...")
    test_problems = generate_math_problems(n_problems=160, min_num=1, max_num=200,
                                           balance_ops=True, multiply_curriculum=False, multiply_weight=1)
    train_problems = generate_math_problems(n_problems=200, min_num=1, max_num=200,
                                            balance_ops=True, multiply_curriculum=True, multiply_weight=2)

    print(f"训练集: {len(train_problems)} 题")
    print(f"测试集: {len(test_problems)} 题")
    print(f"训练集运算符分布: { {op: sum(1 for p in train_problems if p.op == op) for op in ['+', '-', '*', '/']} }")
    print(f"测试集运算符分布: { {op: sum(1 for p in test_problems if p.op == op) for op in ['+', '-', '*', '/']} }")

    print("\n" + "-" * 50)
    print("【阶段1】测试GPRO之前的模型准确率...")
    print("-" * 50)
    baseline_results = test_model_accuracy(model, tokenizer, test_problems, device=device)
    print(f"GPRO之前准确率: {baseline_results['accuracy']:.2%} ({baseline_results['correct']}/{baseline_results['total']})")

    print("\n" + "-" * 50)
    print("【阶段2】构建GPRO训练数据集...")
    print("-" * 50)
    train_dataset = build_gpro_dataset(train_problems)
    print(f"训练数据集大小: {len(train_dataset)}")

    print("\n" + "-" * 50)
    print("【阶段3】使用GPRO强化学习微调模型...")
    print("-" * 50)
    gpro_model = run_gpro_training(model, tokenizer, train_dataset)
    gpro_model.eval()

    print("\n" + "-" * 50)
    print("【阶段4】测试GPRO之后的模型准确率...")
    print("-" * 50)
    gpro_results = test_model_accuracy(gpro_model, tokenizer, test_problems, device=device)
    print(f"GPRO之后准确率: {gpro_results['accuracy']:.2%} ({gpro_results['correct']}/{gpro_results['total']})")

    print_comparison_report(baseline_results, gpro_results)

    output_results = {
        "baseline": {
            "accuracy": baseline_results["accuracy"],
            "correct": baseline_results["correct"],
            "total": baseline_results["total"],
            "op_accuracies": baseline_results["op_accuracies"],
            "op_stats": baseline_results["op_stats"],
        },
        "gpro": {
            "accuracy": gpro_results["accuracy"],
            "correct": gpro_results["correct"],
            "total": gpro_results["total"],
            "op_accuracies": gpro_results["op_accuracies"],
            "op_stats": gpro_results["op_stats"],
        },
        "improvement": {
            "absolute": gpro_results["accuracy"] - baseline_results["accuracy"],
            "relative_pct": (gpro_results["accuracy"] - baseline_results["accuracy"]) / baseline_results["accuracy"] * 100 if baseline_results["accuracy"] > 0 else 0,
        },
        "detailed_results": {
            "baseline": baseline_results["results"],
            "gpro": gpro_results["results"],
        }
    }

    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_results, f, ensure_ascii=False, indent=2)
    print(f"详细结果已保存至: {RESULTS_FILE}")


if __name__ == "__main__":
    main()