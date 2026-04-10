import os
from env import SmartShoppingEnv, Action

# Environment variables (as required by hackathon)
API_BASE_URL = os.getenv("API_BASE_URL", "local")
MODEL_NAME = os.getenv("MODEL_NAME", "rule-based")


# ---------------- SMART POLICY ---------------- #
def smart_policy(product, history):
    score = 0

    # Rating importance
    score += product.rating * 2

    # Budget check
    if product.price <= product.budget:
        score += 3

    # Discount bonus
    if product.discount >= 20:
        score += 2

    # High rating bonus
    if product.rating >= 4:
        score += 2

    # Decision logic
    if score >= 10:
        return "buy"
    elif score >= 7:
        return "compare"
    elif score >= 5:
        return "wishlist"
    else:
        return "skip"


# ---------------- RUN FUNCTION ---------------- #
def run():
    env = SmartShoppingEnv()
    obs = env.reset()

    total_reward = 0
    step = 0
    history = []

    print(f"[START] task=shopping env=SmartShopping model={MODEL_NAME}", flush=True)

    while True:
        product = obs.current_product

        # CORE DECISION LINE
        action = smart_policy(product, history)

        # Environment step
        obs, reward, done, _ = env.step(Action(action_type=action))

        total_reward += reward.value

        print(f"[STEP] step={step} action={action} reward={reward.value:.2f} done={done}", flush=True)

        history.append((product.name, action, reward.value))
        step += 1

        if done:
            break

    print(f"[END] success=true steps={step} score={total_reward:.2f}", flush=True)


# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    run()