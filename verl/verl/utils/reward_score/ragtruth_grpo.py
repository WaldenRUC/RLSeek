import re, ast, tiktoken
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Hashable, Sequence, TypeVar
encoding = tiktoken.get_encoding("cl100k_base")
def count_tokens(text):
    return len(encoding.encode(text))


def find_start_end_index(string: str, sub_str: str, strict=False):
    """
    Get the (start, end) index for span in the response string
    :param string: LLM response string
    :param sub_str: predicted hallucination spans
    :return: (start, end), where string[start:end] is sub_str
    """
    # check whether string and sub_str is STR type
    string = str(string).lower()
    sub_str = str(sub_str).lower().strip()
    start = string.find(sub_str)
    if start == -1: # response中找不到预测的幻觉文本
        if len(sub_str) >= 10 and not strict:
            # Allow matching 10 characters at the beginning and end, with a total length ratio between [0.7, 1.3].
            match_length = 10
            prefix_match = string.find(sub_str[:match_length])
            suffix_match = string.find(sub_str[-match_length:])
            new_start, new_end = prefix_match, suffix_match + match_length
            
            if prefix_match != -1 and suffix_match != -1 and 0.7 < (new_end - new_start)/len(sub_str) < 1.3:
                # 找到头尾了, 而且在范围内, 那么返回模糊匹配的结果
                return new_start, new_end
            else:
                sub_str = ' '.join(sub_str.split())
                start = string.find(sub_str)
                if start == -1:
                    return -1, -1
                else: return (start, start + len(sub_str))
        else:   # 要么Strict, 要么sub_str长度<10, 但还是找不到位置
            return -1, -1
    end = start + len(sub_str)
    return (start, end)


def extract_hallucination_dict(text: str):
    match = re.search(r'\{[^{}]*"hallucination list"[^{}]*\}', text)
    if not match:
        return None  # 没找到
    dict_str = match.group(0)
    try:
        result = ast.literal_eval(dict_str)["hallucination list"]
    except Exception:
        import json
        try:
            result = json.loads(dict_str)["hallucination list"]
        except Exception:
            return None
    return result

    
def compute_multi_spans(gold_spans, pred_spans):
    """
    Compute the F1 score for multiple spans.
    :param gold_spans: A list of ground truth spans, each represented as a (start, end) tuple
    :param pred_spans: A list of predicted spans, each represented as a (start, end) tuple
    :return: The F1 score
    """
    gold_sets = [set(range(int(start), int(end))) for start, end in gold_spans]
    pred_sets = [set(range(int(start), int(end))) for start, end in pred_spans]

    gold_union = set.union(*gold_sets)
    pred_union = set.union(*pred_sets)

    tp = len(gold_union.intersection(pred_union))
    pred_len = len(pred_union)
    gold_len = len(gold_union)

    return tp, pred_len, gold_len
    
def calculate_span_f1(extracted_halu_list: list, ground_truth: str, response: str):
    ground_truth_list = ast.literal_eval(ground_truth)
    predict_span = []
    for extracted_halu in extracted_halu_list:
        start, end = find_start_end_index(response, extracted_halu, strict=False)
        if (start, end) != (-1, -1):
            predict_span.append((start, end))
        else: return 0
    
    ground_truth_span = []
    for label in ground_truth_list:
        ground_truth_span.append((label['start'], label['end']))
        
    if len(predict_span) == 0:
        predict_span.append((0, 0))
    if len(ground_truth_span) == 0:
        ground_truth_span.append((0, 0))
    
    tp, pred_len, gold_len = compute_multi_spans(ground_truth_span, predict_span)
    if pred_len == 0 and gold_len == 0:
        span_precision, span_recall, span_f1 = 1, 1, 1
    # pred_len gold_len不同时为0
    else:
        span_precision = tp / pred_len if pred_len > 0 else 0
        span_recall = tp / gold_len if gold_len > 0 else 0
        span_f1 = (2 * span_precision * span_recall / (span_precision + span_recall)) if (span_precision + span_recall) > 0 else 0
    return span_f1


def lcs_length(a: str, b: str) -> int:
    """
    Compute the character-level LCS length of two strings.
    Case-sensitive. Whitespace and punctuation are treated as normal characters.
    """
    if len(a) < len(b):
        a, b = b, a
    # dp[j] = LCS length of a[:i] and b[:j]
    dp = [0] * (len(b) + 1)
    for i in range(len(a)):
        prev = 0
        for j in range(len(b)):
            tmp = dp[j + 1]
            if a[i] == b[j]:
                dp[j + 1] = prev + 1
            else:
                dp[j + 1] = max(dp[j + 1], dp[j])
            prev = tmp
    return dp[-1]

# def longest_common_substring(a: str, b: str):
#     n, m = len(a), len(b)
#     dp = [0] * (m + 1)
#     best_len = 0
#     end_pos_in_a = 0  # end index (1-based) of best substring in a
#     for i in range(1, n + 1):
#         prev_diag = 0
#         for j in range(1, m + 1):
#             tmp = dp[j]
#             if a[i - 1] == b[j - 1]:
#                 dp[j] = prev_diag + 1
#                 if dp[j] > best_len:
#                     best_len = dp[j]
#                     end_pos_in_a = i
#             else:
#                 dp[j] = 0
#             prev_diag = tmp
#     return best_len
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

def check_step_quote(text):
    pattern = re.compile(
        r"## Step\b(?![\s\S]*?<quote>[\s\S]*?</quote>[\s\S]*?(?=## Step\b|$))"
    )
    some_steps_donot_have_quote = bool(pattern.search(text))
    return some_steps_donot_have_quote

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
    qoute_tokens_length = np.mean(
        [count_tokens(_quote) for _quote in quotes]
    )
    return len(quotes), quote_char_length, qoute_tokens_length, penalty

def compute_score(solution_str, ground_truth, extra_info, add_penalty=True) -> float:
    #& format checking
    extracted_halu_list = extract_hallucination_dict(solution_str)
    if extracted_halu_list == None:
        return {
            'score': -1,
            'span_f1': -1,
            'penalty': -1,
            'quote_num': -1, # quote的数量
            'quote_char_length': -1,    # 平均每个quote内的character数量
            'quote_tokens': -1,
            'format_error': -1,
        }
    if not isinstance(extracted_halu_list, list):
        {
            'score': -1,
            'span_f1': -1,
            'penalty': -1,
            'quote_num': -1, # quote的数量
            'quote_char_length': -1,    # 平均每个quote内的character数量
            'quote_tokens': -1,
            'format_error': -1,
        }
    
    #& 计算reward r
    response = extra_info['response']
    reward = calculate_span_f1(extracted_halu_list, ground_truth, response)
    
    #& 计算penalty r_err
    if add_penalty == False:
        # 不考虑penalty
        quote_num, quote_char_length, quote_token_length, r_penalty = 0, 0, 0, 0
    else:
        quote_num, quote_char_length, quote_token_length, r_penalty = calculate_penalty(
            solution_str, extra_info['reference'])
    
    return {
        'score': reward - r_penalty,
        'span_f1': reward,
        'penalty': r_penalty,
        'quote_num': quote_num,
        'quote_char_length': quote_char_length,    # 平均每个quote内的character数量
        'quote_tokens': quote_token_length,
        'format_error': 0
    }




def calculate_span(extracted_halu_list: list, ground_truth: str, response: str):
    ground_truth_list = ast.literal_eval(ground_truth)  # 每个元素表示一个幻觉片段
    predict_span = []
    for extracted_halu in extracted_halu_list:
        start, end = find_start_end_index(response, extracted_halu, strict=False)
        if (start, end) != (-1, -1):
            predict_span.append((start, end))
    
    ground_truth_span = []
    for label in ground_truth_list:
        ground_truth_span.append((label['start'], label['end']))
        
    if len(predict_span) == 0:
        predict_span.append((0, 0))
    if len(ground_truth_span) == 0:
        ground_truth_span.append((0, 0))
    
    tp, pred_len, gold_len = compute_multi_spans(ground_truth_span, predict_span)
    if pred_len == 0 and gold_len == 0:
        span_precision, span_recall, span_f1 = 1, 1, 1
    else:
        span_precision = tp / pred_len if pred_len > 0 else 0
        span_recall = tp / gold_len if gold_len > 0 else 0
        span_f1 = (2 * span_precision * span_recall / (span_precision + span_recall)) if (span_precision + span_recall) > 0 else 0
    if predict_span == [(0,0)]: pred_claim = 0
    else: pred_claim = 1
    if ground_truth_span == [(0,0)]: gt_claim = 0
    else: gt_claim = 1
    return {
        'claim': {
            'pred': pred_claim,
            'gt': gt_claim
        },
        'span': {   # 对所有样本取平均值
            'precision': span_precision,
            'recall': span_recall,
            'f1': span_f1
        }
    }

def compute_score_metric(solution_str: str, ground_truth: str, response) -> float:
    extracted_halu_list = extract_hallucination_dict(solution_str)
    if extracted_halu_list == None:
        return 0
    if not isinstance(extracted_halu_list, list):
        return 0
    reward_dict = calculate_span(extracted_halu_list, ground_truth, response)
    return reward_dict

if __name__ == '__main__':
    pass