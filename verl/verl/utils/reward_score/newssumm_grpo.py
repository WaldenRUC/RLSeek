import re, ast
from tracemalloc import get_traceback_limit

from evaluate import format_error
def lcs_length(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = [ [0]*(n+1) for _ in range(m+1) ]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
def span_reward(gold_spans: list, pred_spans: list):
    """
    Maximum matching + LCS char-level precision/recall/f1
    TP = sum of LCS lengths for matched spans
    Precision = TP / total number of predicted characters
    Recall    = TP / total number of gold characters
    """
    # Neg samples w/o span reward
    if len(gold_spans) == len(pred_spans) == 0:
        return 1.0, 1.0, 1.0
    if len(gold_spans) == 0:
        return 0.0, 0.0, 0.0
    if len(pred_spans) == 0:
        return 0.0, 0.0, 0.0

    edges = []
    for i, p in enumerate(pred_spans):
        for j, g in enumerate(gold_spans):
            overlap = lcs_length(p, g)
            if overlap > 0:
                edges.append((overlap, i, j, p, g))

    # Sort edges by LCS length in descending order for greedy matching
    edges.sort(reverse=True, key=lambda x: x[0])

    matched_pred = set()
    matched_gold = set()
    tp_chars = 0

    for overlap, i, j, p, g in edges:
        if i not in matched_pred and j not in matched_gold:
            matched_pred.add(i)
            matched_gold.add(j)
            tp_chars += overlap

    total_pred_chars = sum(len(p) for p in pred_spans)
    total_gold_chars = sum(len(g) for g in gold_spans)

    precision = tp_chars / total_pred_chars if total_pred_chars > 0 else 0.0
    recall = tp_chars / total_gold_chars if total_gold_chars > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall

def format_reward(solution_str: str) -> float:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<answer>.*?</answer>"
    match = re.match(pattern, solution_str, re.DOTALL)
    return 1.0 if match else -1.0

def extract_answers(text):
    return re.findall(r'<answer>(.*?)</answer>', text, flags=re.DOTALL)

def compute_score(solution_str, ground_truth, extra_info, add_penalty=False) -> float:
    gt = ast.literal_eval(ground_truth)
    invalid_return = {
        'score': -1,
        'span_f1': -1,
        'span_precision': -1,
        'span_recall': -1,
        'penalty': -1,
        'quote_num': -1, # quote的数量
        'quote_char_length': -1,    # 平均每个quote内的character数量
        'quote_tokens': -1,
        'format_error': -1,
    }
    answers = extract_answers(solution_str)
    if len(answers) != 1:
        return invalid_return
    
    answers = answers[0]
    try:
        spans_list = ast.literal_eval(answers)
    except Exception:
        return invalid_return
    if not isinstance(spans_list, list):
        return invalid_return
    for _ in spans_list:
        if not isinstance(_, str):
            return invalid_return
    
    span_f1, span_precision, span_recall = span_reward(gt, spans_list)
    return {
        'score': span_f1,
        'span_f1': span_f1,
        'span_precision': span_precision,
        'span_recall': span_recall,
        'penalty': 0,
        'quote_num': 0, # quote的数量
        'quote_char_length': 0,    # 平均每个quote内的character数量
        'quote_tokens': 0,
        'format_error': 0,
    }


def span_reward_chars(gold_spans: list, pred_spans: list):
    """
    Maximum matching + LCS char-level precision/recall/f1
    TP = sum of LCS lengths for matched spans
    Precision = TP / total number of predicted characters
    Recall    = TP / total number of gold characters
    """
    # Neg samples w/o span reward
    tp_chars = 0
    total_pred_chars = sum(len(p) for p in pred_spans)
    total_gold_chars = sum(len(g) for g in gold_spans)

    edges = []
    for i, p in enumerate(pred_spans):
        for j, g in enumerate(gold_spans):
            overlap = lcs_length(p, g)
            if overlap > 0:
                edges.append((overlap, i, j, p, g))

    # Sort edges by LCS length in descending order for greedy matching
    edges.sort(reverse=True, key=lambda x: x[0])

    matched_pred = set()
    matched_gold = set()
    

    for overlap, i, j, p, g in edges:
        if i not in matched_pred and j not in matched_gold:
            matched_pred.add(i)
            matched_gold.add(j)
            tp_chars += overlap

    

    # precision = tp_chars / total_pred_chars if total_pred_chars > 0 else 0.0
    # recall = tp_chars / total_gold_chars if total_gold_chars > 0 else 0
    # f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return tp_chars, total_pred_chars, total_gold_chars

def compute_score_metric(solution_str, ground_truth):
    gt = ast.literal_eval(ground_truth)
    format_error_flag = 0
    answers = extract_answers(solution_str)
    if len(answers) != 1:
        answers = [[]]
        format_error_flag = 1
    
    answers = answers[0]
    try:
        answers = ast.literal_eval(answers)
    except Exception:
        answers = []
        format_error_flag = 1
    if not isinstance(answers, list):
        answers = []
        format_error_flag = 1
    for _ in answers:
        if not isinstance(_, str):
            answers = []
            format_error_flag = 1
    
    tp_chars, total_pred_chars, total_gold_chars = span_reward_chars(gt, answers)
    if gt == []: gt_claim = 0
    else: gt_claim = 1
    if answers == []: pred_claim = 0
    else: pred_claim = 1
    reward_dict = {
        'claim': {
            'pred': pred_claim,
            'gt': gt_claim
        },
        'span': {
            'tp': tp_chars,
            'pred_len': total_pred_chars,
            'gold_len': total_gold_chars
        }
    }
    return format_error_flag, reward_dict