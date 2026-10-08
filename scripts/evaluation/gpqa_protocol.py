import itertools
import re


def record(identifier, prompt, target, *, question_id=None, group="", completion=False, **metadata):
    return {
        "id": str(identifier),
        "question_id": str(question_id if question_id is not None else identifier),
        "prompt": prompt,
        "target": target,
        "group": group,
        "completion": completion,
        "metadata": metadata,
    }


def mcq_prompt(question, options):
    choices = "\n".join(f"({chr(65 + i)}) {text}" for i, text in enumerate(options))
    return (
        f"What is the correct answer to this question?\n{question}\n\nChoices:\n{choices}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}. "
        "Include only the correct option letter inside the box."
    )


def gpqa_samples(rows):
    for i, row in enumerate(rows):
        question_id = row.get("Record ID") or str(i)

        options = [
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
            row["Correct Answer"],
        ]
        for permutation_id, permutation in enumerate(itertools.permutations(range(4))):
            correct_position = permutation.index(3)
            yield record(
                f"{question_id}/p{permutation_id:02}",
                mcq_prompt(row["Question"], [options[j] for j in permutation]),
                chr(65 + correct_position),
                question_id=question_id,
                source_row=i,
                permutation_id=permutation_id,
                permutation=list(permutation),
                original_correct_option_id=3,
                correct_position=correct_position,
                num_choices=4,
            )


def last_boxed(text):
    starts = list(re.finditer(r"\\boxed\s*\{", text))
    if not starts:
        return None
    start = starts[-1].end()
    depth = 1
    for i in range(start, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return text[start:i].strip()
    return None


def answer_text(text):
    text = re.sub(r"\*\*|__", "", text)
    return re.sub(r"(?m)^\s*#{1,6}\s+", "", text)


def normalize_answer_surface(value):
    value = value.strip()
    for _ in range(12):
        before = value
        for left, right in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
            if value.startswith(left) and value.endswith(right) and len(value) >= len(left) + len(right):
                value = value[len(left) : -len(right)].strip()
                break
        match = re.match(r"^\\(?:text|mathrm|mathbf|mathit|textbf|mbox|operatorname)\s*\{", value)
        if match:
            depth, end = 1, match.end()
            while end < len(value) and depth:
                depth += (value[end] == "{") - (value[end] == "}")
                end += 1
            if depth == 0 and not value[end:].strip():
                value = value[match.end() : end - 1].strip()
        if value == before:
            break
    return " ".join(value.split())


def prompt_choices(prompt, num_choices):
    if "\nChoices:\n" not in prompt:
        return []
    block = prompt.rsplit("\nChoices:\n", 1)[1].split("\n\nPlease reason step by step,", 1)[0]
    labels = list(re.finditer(r"(?m)^\(([A-J])\) ", block))
    if [m.group(1) for m in labels] != [chr(65 + i) for i in range(num_choices)]:
        return []
    return [
        block[m.end() : labels[i + 1].start() if i + 1 < len(labels) else len(block)].strip()
        for i, m in enumerate(labels)
    ]


def extract_mcq(text, num_choices, options=None):
    prediction = _extract_mcq_existing(text, num_choices)
    if prediction is not None:
        return prediction
    clean = answer_text(text)
    boxed = last_boxed(clean)

    later = clean[clean.rfind("\\boxed") :] if boxed is not None else ""
    if boxed is not None and re.search(
        r"(?:final|correct|best)\s+(?:answer|option|choice)|the\s+answer\s+is", later, re.I
    ):
        return None
    candidate = boxed if boxed is not None else clean.strip()
    if boxed is None:
        markers = list(
            re.finditer(
                r"(?:the\s+(?:(?:correct|best)\s+)?answer\s+is|(?:final|correct|best)\s+(?:answer|option|choice)(?:\s+is)?\s*[:：]?|answer\s*[:：])\s*",
                clean,
                re.I,
            )
        )
        if markers:
            candidate = clean[markers[-1].end() :].strip()
    normalized = normalize_answer_surface(candidate)

    label = re.fullmatch(r"(?:\(([A-J])\)|([A-J]))[.。,:：;；!！]?", normalized)
    if label:
        letter = label.group(1) or label.group(2)
        return letter if ord(letter) - 65 < num_choices else None
    if boxed is not None and options is not None and len(options) == num_choices:
        def surface(value):
            return normalize_answer_surface(value).removesuffix("。").removesuffix(".").strip()

        original_boxed = last_boxed(text)
        hits = [i for i, option in enumerate(options) if surface(option) == surface(original_boxed)]
        if len(hits) == 1:
            return chr(65 + hits[0])
    return None


def _extract_mcq_existing(text, num_choices):
    text = answer_text(text)
    boxed = last_boxed(text)
    matches = list(
        re.finditer(
            r"(?:the\s+(?:(?:correct|best)\s+)?(?:answer|option|choice)\s+is|(?:final|correct|best)\s+(?:answer|option|choice)(?:\s+is)?\s*[:：]?|answer\s*[:：])\s*",
            text,
            re.I,
        )
    )
    candidate = text[matches[-1].end() :].split("\n", 1)[0].strip() if matches else text.strip()
    if boxed is not None and (not matches or text.rfind("\\boxed") > matches[-1].start()):
        candidate = boxed
    candidate = re.sub(r"^\\(?:text|mathrm|mathbf)\s*\{(.*?)\}$", r"\1", candidate.strip()).strip("* ")
    match = re.fullmatch(r"\(?([A-J])\)?[.。]?", candidate.strip(), re.I)
    if not match and (matches or boxed is not None):
        match = re.match(r"^\(([A-J])\)(?=\s|[.,]|$)", candidate)
        if not match:
            match = re.match(r"^([A-J])(?=\s|[.,]|$)", candidate)
        if match:
            rest = candidate[match.end() :]
            if re.search(r"\([A-J]\)|\b(?:or|and)\s+[A-J]\b|^\s*[,/]\s*[A-J]\b", rest, re.I):
                match = None
    return match.group(1).upper() if match and ord(match.group(1).upper()) - 65 < num_choices else None
