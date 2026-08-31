import os
from typing import Optional

import gradio as gr
from core.chatbot_tool_use import SeismicChatBotToolUse
from core.skills import CONTEXT_PARAMS, capture_skill, get_registry
from core.trace_summary import format_trace_markdown, summarize_trace
from config.example_prompts import EXAMPLE_PROMPTS, search_prompts, get_random_prompts
from config.settings import SEISMIC_UPLOAD_DIR, MAX_IMAGE_MB
from tools.image_safety import stage_upload

import logging
from config.settings import LOG_LEVEL, LOG_FORMAT

# No-op when main.py already configured the root logger; makes direct imports
# / uvicorn-style launches emit logs instead of silence.
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format=LOG_FORMAT)


def append_bot_response(chat_history, response):
    """Append a bot response to Gradio 3.x pair-format chat history.

    The tool-use bot returns {"reply": str, "images": list[str]}: the reply
    fills the pending assistant slot, then each image gets its own history row
    (Gradio renders one file per message). Plain strings render as-is.
    """
    if isinstance(response, dict) and "reply" in response:
        reply = response.get("reply") or ""
        trace = response.get("trace")
        if isinstance(trace, dict) and trace.get("events"):
            try:
                flags = summarize_trace(trace)["flags"]
            except Exception as e:
                logging.getLogger(__name__).warning(f"trace summary failed: {e}")
                flags = []
            if flags:
                reply = (reply + "\n\n" + "\n".join(flags)).strip()
        chat_history[-1][1] = reply
        for path in response.get("images") or []:
            chat_history.append([None, (path,)])
    # Defensive: process_single_input always returns the dict above; these
    # branches keep plain-string/legacy responses renderable (unit-tested).
    elif isinstance(response, str):
        chat_history[-1][1] = response
    else:
        chat_history[-1][1] = str(response)
    return chat_history


def format_status(token_usage, trace=None):
    """One-line session status: token totals plus the turn's tool chain."""
    status = (f"Prompt: {token_usage['prompt_tokens']} | "
              f"Completion: {token_usage['completion_tokens']} | "
              f"Total: {token_usage['total_tokens']}")
    tools = (trace or {}).get("tools_used") or []
    if tools:
        status += " | Tools: " + " → ".join(tools)
    return status


def parse_parameter_lines(text: str) -> dict:
    """'slot=value' per line → {slot: typed value}; numbers become int/float."""
    params = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"expected 'name=value', got {line!r}")
        key, value = (part.strip() for part in line.split("=", 1))
        try:
            params[key] = int(value)
        except ValueError:
            try:
                params[key] = float(value)
            except ValueError:
                params[key] = value
    return params


def save_skill_from_ui(session_bot, name, description, params_text) -> str:
    """Save the most recent completed turn as a skill; returns status markdown."""
    if session_bot is None:
        return "⚠️ Send a message first — there is no turn to save yet."
    try:
        cm = session_bot.context_manager
        calls = cm.get_context("current_turn_calls") or []
        data = capture_skill((name or "").strip(), (description or "").strip(),
                             parse_parameter_lines(params_text), calls,
                             cm.get_context("current_turn_input") or "",
                             set(CONTEXT_PARAMS))
        registry = get_registry()
        path = registry.save(data)
        index = getattr(session_bot, "tool_index", None)
        if index is not None and hasattr(index, "refresh"):
            index.refresh(registry.specs())
        return f"✅ Saved skill **{name}** ({len(data['chain'])} step(s)) → `{path}`"
    except Exception as e:
        return f"⚠️ Could not save skill: {e}"


def skills_markdown() -> str:
    skills = get_registry().list()
    if not skills:
        return "_No skills saved yet._"
    lines = []
    for s in skills:
        params = ", ".join(f"{p}={sch.get('default', '?')}" for p, sch in s["parameters"].items())
        kind = "replay" if s["has_chain"] else "guided"
        lines.append(f"- **{s['name']}** ({kind}, {s['source']}) — {s['description']}  \n  parameters: {params or 'none'}")
    return "\n".join(lines)


DEFAULT_IMAGE_REQUEST = "Interpret this outcrop photo."


def prepare_turn(message: str, image_path: Optional[str], session_bot,
                 upload_dir: str, max_mb: float) -> str:
    """Stage an uploaded photo into the session sandbox and mark the message.

    The staged path is stored on the session (context key ``last_image``) so
    the outcrop tools can pick it up without the LLM ever handling a path.
    Returns the text to send to the chatbot.
    """
    message = (message or "").strip()
    if not image_path:
        return message
    staged = stage_upload(image_path, upload_dir, session_bot.session_id, max_mb)
    session_bot.attach_image(staged)
    if not message:
        message = DEFAULT_IMAGE_REQUEST
    return f"[image attached: {os.path.basename(staged)}] {message}"


def create_chat_interface(base_bot=None):
    """Create and return the Gradio chat interface using the tool use pattern.

    base_bot: any object with new_session()/process_single_input()/attach_image()
    (SeismicChatBotToolUse or SeismicOrchestrator). Default: the classic bot.
    """
    # Build the heavy, conversation-stateless components ONCE. Each browser
    # session gets its own isolated chatbot (fresh context + token counter) via
    # new_session(), held in gr.State so users never share conversation state.
    if base_bot is None:
        base_bot = SeismicChatBotToolUse()

    def respond(message, image_path, chat_history, session_bot):
        """Process a user message (+ optional photo) using a per-session chatbot."""
        if session_bot is None:
            session_bot = base_bot.new_session()

        chat_history = chat_history or []
        shown = message or ("(photo uploaded)" if image_path else "")
        chat_history.append([shown, None])
        try:
            text = prepare_turn(message, image_path, session_bot, SEISMIC_UPLOAD_DIR, MAX_IMAGE_MB)
            response = session_bot.process_single_input(text)
            chat_history = append_bot_response(chat_history, response)

            # Per-session token usage for display
            token_usage = session_bot.context_manager.get_token_usage()
            trace = response.get("trace") if isinstance(response, dict) else None
            token_str = format_status(token_usage, trace)
            try:
                trace_md = format_trace_markdown(trace)
            except Exception as e:
                logging.getLogger(__name__).warning(f"trace markdown failed: {e}")
                trace_md = "_No decision trace for this turn._"
            return "", None, chat_history, token_str, trace_md, session_bot

        except Exception as e:
            chat_history[-1][1] = f"Error processing request: {str(e)}"
            token_str = format_status(session_bot.context_manager.get_token_usage())
            return "", None, chat_history, token_str, format_trace_markdown(None), session_bot
    
    def copy_prompt(prompt_text):
        """Copy prompt to clipboard and return it for the textbox."""
        return prompt_text
    
    def search_examples(query):
        """Search for example prompts."""
        if not query.strip():
            results = get_random_prompts(5)
        else:
            results = search_prompts(query)
        
        # Format results as HTML
        if not results:
            return "<p><em>No examples found. Try a different search term.</em></p>"
        
        html_content = "<div style='max-height: 300px; overflow-y: auto;'>"
        for i, result in enumerate(results):
            html_content += f"""
            <div style='margin-bottom: 15px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>
                <h4 style='margin: 0 0 5px 0; color: #333;'>{result['title']}</h4>
                <p style='margin: 0 0 8px 0; color: #666; font-style: italic;'>{result['description']}</p>
                <div style='background-color: white; padding: 8px; border-radius: 3px; border: 1px solid #ccc;'>
                    <code style='color: #333;'>{result['prompt']}</code>
                </div>
            </div>
            """
        html_content += "</div>"
        return html_content
    
    with gr.Blocks(title="Seismic Modeling Assistant - Tool Use") as demo:
        # Per-session chatbot, isolated per browser connection.
        session_state = gr.State(None)

        gr.Markdown("""
        # 🌊 Seismic Modeling Assistant (Tool Use Pattern)
        
        Welcome to the Seismic Modeling Assistant! I can help you with:
        
        - Creating and analyzing Ricker wavelets
        - Building wedge models for seismic analysis
        - Calculating AVO reflectivity using Zoeppritz and Shuey equations
        - Answering questions about seismic properties
        - Explaining seismic modeling concepts
        - Interpreting an uploaded **outcrop photo** into a 2-D earth model and synthetic seismic section

        **💡 Tip:** Use the example prompts below to get started quickly!
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                chat_display = gr.Chatbot(height=600)
                photo = gr.Image(type="filepath", label="Outcrop photo (optional — jpg/png/webp)",
                                 height=160)
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask a question or request an action...",
                        show_label=False,
                        container=False
                    )
                    submit = gr.Button("Send", variant="primary")
                # Add token usage display with custom styling
                token_usage_display = gr.Markdown(
                    "Token Usage: 0", 
                    elem_id="token-usage",
                    elem_classes=["token-usage-display"]
                )
                
                # Add custom CSS for token usage display
                gr.HTML("""
                <style>
                .token-usage-display {
                    background-color: #f0f7ff;
                    border: 1px solid #cce5ff;
                    border-radius: 5px;
                    padding: 8px 12px;
                    margin-top: 10px;
                    font-size: 0.9em;
                    color: #0366d6;
                    font-family: monospace;
                    text-align: center;
                }
                </style>
                """)

                with gr.Accordion("🔍 Decision trace (last turn)", open=False):
                    trace_display = gr.Markdown("_No decision trace yet._")

                with gr.Accordion("🧩 Skills", open=False):
                    skills_display = gr.Markdown(skills_markdown())
                    with gr.Row():
                        skill_name = gr.Textbox(label="Name", placeholder="tuning_from_petro", scale=2)
                        skill_desc = gr.Textbox(label="Description", placeholder="What it does", scale=3)
                    skill_params = gr.Textbox(label="Parameters (one per line: name=value used last turn)",
                                              placeholder="freq=30\nphit=0.25", lines=3)
                    save_btn = gr.Button("💾 Save last turn as skill")
                    save_status = gr.Markdown("")

            with gr.Column(scale=2):
                # Example prompts section
                gr.Markdown("### 📋 Quick Examples")
                
                # Create accordion with categorized examples
                with gr.Accordion("📋 Example Prompts", open=False):
                    # Search box
                    search_box = gr.Textbox(
                        placeholder="Search examples...",
                        label="Search Examples",
                        scale=4
                    )
                    
                    # Search results - initialize with some default examples
                    default_results = get_random_prompts(3)
                    default_html = "<div style='max-height: 300px; overflow-y: auto;'>"
                    for result in default_results:
                        default_html += f"""
                        <div style='margin-bottom: 15px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>
                            <h4 style='margin: 0 0 5px 0; color: #333;'>{result['title']}</h4>
                            <p style='margin: 0 0 8px 0; color: #666; font-style: italic;'>{result['description']}</p>
                            <div style='background-color: white; padding: 8px; border-radius: 3px; border: 1px solid #ccc;'>
                                <code style='color: #333;'>{result['prompt']}</code>
                            </div>
                        </div>
                        """
                    default_html += "</div>"
                    
                    search_results = gr.HTML(
                        value=default_html,
                        label="Search Results"
                    )
                    
                    # Create accordion for each category
                    for category, prompts in EXAMPLE_PROMPTS.items():
                        with gr.Accordion(category, open=False):
                            for prompt in prompts:
                                with gr.Column():
                                    gr.Markdown(f"**{prompt['title']}**")
                                    gr.Markdown(f"*{prompt['description']}*")
                                    
                                    # Prompt text with small copy button inline
                                    with gr.Row():
                                        prompt_text = gr.Textbox(
                                            value=prompt['prompt'],
                                            label="",
                                            interactive=False,
                                            container=False,
                                            scale=8
                                        )
                                        copy_btn = gr.Button(
                                            "📋",
                                            size="sm",
                                            variant="secondary",
                                            scale=1,
                                            min_width=30
                                        )
                                        # Connect copy button to main textbox
                                        copy_btn.click(
                                            fn=lambda x: x,
                                            inputs=prompt_text,
                                            outputs=msg
                                        )
                    
                    # Connect search functionality
                    search_box.change(
                        search_examples,
                        inputs=[search_box],
                        outputs=[search_results]
                    )
                
                # Quick tips
                gr.Markdown("""
                ### 💡 Quick Tips
                
                **Educational Questions:**
                - Ask about seismic concepts and theory
                - Get explanations of modeling techniques
                - Learn about interpretation methods
                
                **Tool Actions:**
                - Use natural language to specify parameters
                - The AI will extract parameters automatically
                - You can be as detailed or brief as you prefer
                
                **Examples:**
                - "Create a 30 Hz Ricker wavelet"
                - "Make a wedge model with 100m thickness"
                - "Calculate Zoeppritz reflectivity for gas sand"
                """)
        
        submit.click(respond, [msg, photo, chat_display, session_state],
                     [msg, photo, chat_display, token_usage_display, trace_display, session_state])
        msg.submit(respond, [msg, photo, chat_display, session_state],
                   [msg, photo, chat_display, token_usage_display, trace_display, session_state])

        def on_save_skill(name, description, params_text, session_bot):
            status = save_skill_from_ui(session_bot, name, description, params_text)
            return status, skills_markdown()

        save_btn.click(on_save_skill, [skill_name, skill_desc, skill_params, session_state],
                       [save_status, skills_display])

    return demo

if __name__ == "__main__":
    demo = create_chat_interface()
    demo.launch()
