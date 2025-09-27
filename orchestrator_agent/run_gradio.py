# app.py
import os, asyncio, json
import gradio as gr

from orchestrator_agent import orchestrator_agent, ensure_mcp, close_tools  # import your functions

# (Windows often benefits from this to avoid selector quirks)
if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# --- Gradio handlers ---

async def on_start():
    # Connect MCP tools once, in Gradio's event loop
    await ensure_mcp()

async def on_shutdown():
    # Optional: clean shutdown. If this ever causes loop issues, you can skip closing.
    await close_tools()

async def chat_respond(message: str, history: list[str]):
    """
    Gradio will pass (message, history). We only need message.
    Return a string for the chatbot to display.
    """
    # Call your orchestrator; make sure it does NOT close tools per request
    reply = await orchestrator_agent(message)
    return reply

with gr.Blocks(title="Finance Seer") as demo:
    gr.Markdown("## 💸 Finance Seer\nAsk for a budget or stock help. I’ll route to the right agent.")
    gr.ChatInterface(
        fn=chat_respond,                # async OK
        multimodal=False,
        fill_height=True,
        textbox=gr.Textbox(placeholder="e.g., I live in NYC and make $5,000 net. Make me an entertainment-heavy budget."),
        submit_btn=None,
        stop_btn=None,
    )
    # Connect/disconnect hooks
    demo.load(on_start, queue=False)    # async supported
    demo.unload(on_shutdown)            # async supported in recent Gradio

# Serve
if __name__ == "__main__":
    # queue() enables concurrency + streaming; launch binds the server
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, show_error=True)