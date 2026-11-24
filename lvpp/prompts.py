"""Prompts used by each generator configuration in the paper."""
from __future__ import annotations

import re

# Four worked exemplars, identical to the main-draw and prefix-draw captures.
FEWSHOT = """Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: In April Natalia sold 48 clips. In May she sold half as many, 48 / 2 = 24 clips. Altogether she sold 48 + 24 = 72 clips. The answer is 72.

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Weng earns 12 / 60 = 0.2 dollars per minute. For 50 minutes she earned 50 * 0.2 = 10 dollars. The answer is 10.

Question: Betty is saving money for a new wallet which costs $100. She has only half of the money she needs. Her parents decided to give her $15, and her grandparents twice as much as her parents. How much more money does Betty need?
Answer: Betty has 100 / 2 = 50 dollars. Her parents give 15. Her grandparents give 2 * 15 = 30. Now she has 50 + 15 + 30 = 95 dollars. She needs 100 - 95 = 5 dollars. The answer is 5.

Question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
Answer: The robe takes 2 bolts of blue fiber and 2 / 2 = 1 bolt of white fiber. In total it takes 2 + 1 = 3 bolts. The answer is 3.

"""


def build_fewshot(question: str) -> str:
    return FEWSHOT + f"Question: {question}\nAnswer:"


CHAT_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."


def build_chat(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": CHAT_SYSTEM}, {"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )


# Final-answer triggers used to split reasoning from the answer span.
FEWSHOT_TRIGGER = re.compile(r"[Tt]he answer is")

# The trigger used by the original span-ablation capture (Fig. 2). Note that
# ``\\\\boxed`` matches TWO literal backslashes, so it never fires on a real
# ``\boxed{`` and only "The answer is" splits the solution. Candidates without a
# match get reasoning == full solution and answer span == last token. Kept so the
# published numbers reproduce exactly; see notebooks/04_fig2_span_ablation.ipynb.
CHAT_TRIGGER_LEGACY = re.compile(r"\\\\boxed\{|[Tt]he answer is")

# Corrected trigger: a single literal backslash, and "The final answer is".
CHAT_TRIGGER = re.compile(r"\\boxed\{|[Tt]he (?:final )?answer is")


def split_solution(solution: str, trigger: re.Pattern, cut_next_question: bool = True):
    """Return ``(reasoning_prefix, has_trigger)``; the prefix ends before the last trigger."""
    seg = solution.split("Question:")[0] if cut_next_question else solution
    m = None
    for m in trigger.finditer(seg):
        pass
    return (seg, False) if m is None else (seg[: m.start()], True)
