def grade_easy(history):
    score = 0
    for h in history:
        if h["action"] == "buy" and h["reward"] > 2:
            score += 1
    return min(score / len(history), 1.0)


def grade_medium(history):
    total = sum(h["reward"] for h in history)
    return min(total / 15, 1.0)


def grade_hard(history):
    good_decisions = sum(1 for h in history if h["reward"] > 1)
    return min(good_decisions / len(history), 1.0)