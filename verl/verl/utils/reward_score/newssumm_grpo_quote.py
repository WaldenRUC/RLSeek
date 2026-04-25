import re, ast, tiktoken
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Hashable, Sequence, TypeVar
encoding = tiktoken.get_encoding("cl100k_base")
def count_tokens(text):
    return len(encoding.encode(text))

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

def check_step_quote(text):
    pattern = re.compile(
        r"## Step\b(?![\s\S]*?<quote>[\s\S]*?</quote>[\s\S]*?(?=## Step\b|$))"
    )
    some_steps_donot_have_quote = bool(pattern.search(text))
    return some_steps_donot_have_quote

T = TypeVar("T", bound=Hashable)

@dataclass
class _State:
    next: Dict[T, int] = field(default_factory=dict)
    link: int = -1
    length: int = 0

class SuffixAutomaton:
    def __init__(self):
        self.st: List[_State] = [_State()]  # state 0
        self.last: int = 0

    def extend(self, c: T) -> None:
        st = self.st
        cur = len(st)
        st.append(_State(length=st[self.last].length + 1))
        p = self.last

        while p != -1 and c not in st[p].next:
            st[p].next[c] = cur
            p = st[p].link

        if p == -1:
            st[cur].link = 0
        else:
            q = st[p].next[c]
            if st[p].length + 1 == st[q].length:
                st[cur].link = q
            else:
                clone = len(st)
                st.append(_State(next=st[q].next.copy(), link=st[q].link, length=st[p].length + 1))
                while p != -1 and st[p].next.get(c) == q:
                    st[p].next[c] = clone
                    p = st[p].link
                st[q].link = st[cur].link = clone

        self.last = cur

    def build(self, s: Sequence[T]) -> None:
        for x in s:
            self.extend(x)

    def lcs_len_with(self, t: Sequence[T]) -> int:
        v = 0  # current state
        l = 0  # current match length
        best = 0

        for x in t:
            if x in self.st[v].next:
                v = self.st[v].next[x]
                l += 1
            else:
                while v != -1 and x not in self.st[v].next:
                    v = self.st[v].link
                if v == -1:
                    v = 0
                    l = 0
                    continue
                l = self.st[v].length + 1
                v = self.st[v].next[x]

            if l > best:
                best = l

        return best

def longest_common_substring(a: str, b: str) -> int:
    if not a or not b:
        return 0
    sam = SuffixAutomaton()
    sam.build(a)          # O(n)
    return sam.lcs_len_with(b)  # O(m)

def calculate_penalty(solution_str, reference):
    # 返回quote长度, penalty
    # return: quote length, quote_char_length, quote_token_length, penalty
    pattern = r"<quote>\s*(.*?)\s*</quote>"
    quote_group = re.findall(
        pattern, solution_str, flags=re.DOTALL)
    quotes = [m.strip() for m in quote_group]
    if len(quotes) == 0:
        return 0, 0, 0, 0.5
    if check_step_quote(solution_str):
        return 0, 0, 0, 0.5
    penalty_list = []
    for quote in quotes:
        if quote == '':
            return 0, 0, 0, 0.5
        # lcs = lcs_length(quote, reference)
        lcs = longest_common_substring(quote, reference)
        lcs_ratio = 1 - lcs / len(quote)
        penalty_list.append(max(0, lcs_ratio))
    penalty = np.mean(penalty_list)
    quote_char_length = np.mean(
        [len(_quote) for _quote in quotes]
    )
    quote_tokens_length = np.mean(
        [count_tokens(_quote) for _quote in quotes]
    )
    return len(quotes), quote_char_length, quote_tokens_length, penalty


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
    
    if add_penalty == False:
        quote_num, quote_char_length, quote_token_length, r_penalty = 0, 0, 0, 0
    else:
        quote_num, quote_char_length, quote_token_length, r_penalty = calculate_penalty(
            solution_str, extra_info['source'])
    
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
        'score': span_f1 - r_penalty,
        'span_f1': span_f1,
        'span_precision': span_precision,
        'span_recall': span_recall,
        'penalty': r_penalty,
        'quote_num': quote_num, # quote的数量
        'quote_char_length': quote_char_length,    # 平均每个quote内的character数量
        'quote_tokens': quote_token_length,
        'format_error': 0,
    }