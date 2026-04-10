import gradio as gr
from env import SmartShoppingEnv, Action
from inference import smart_policy


def run_simulation():
    env = SmartShoppingEnv()
    obs = env.reset()

    history = []
    logs = []
    total_reward = 0
    step = 0

    while True:
        product = obs.current_product

        action = smart_policy(product, history)
        obs, reward, done, _ = env.step(Action(action_type=action))

        total_reward += reward.value

        log = f"🛍️ Product: {product.name} | ⭐ {product.rating} | 💰 ₹{product.price} → Action: {action.upper()} | Reward: {reward.value}"
        logs.append(log)

        history.append({
            "product": product.name,
            "action": action,
            "reward": reward.value
        })

        step += 1

        if done:
            break

    final_output = "\n".join(logs)
    summary = f"✅ Total Steps: {step}\n🏆 Final Score: {round(total_reward,2)}"

    return final_output, summary


# ---------------- UI DESIGN ---------------- #

with gr.Blocks(theme=gr.themes.Soft()) as app:

    gr.Markdown("""
    # 🛒 Smart Shopping Assistant
    ### AI-powered decision-making for smarter purchases 💡
    """)

    with gr.Row():
        run_btn = gr.Button("🚀 Run Simulation", variant="primary")

    with gr.Row():
        output_box = gr.Textbox(label="📜 Decision Logs", lines=15)
        summary_box = gr.Textbox(label="📊 Summary", lines=3)

    run_btn.click(fn=run_simulation, outputs=[output_box, summary_box])


# Launch
if __name__ == "__main__":
    app.launch()