import gradio as gr
from core.chatbot_tool_use import SeismicChatBotToolUse
from config.example_prompts import EXAMPLE_PROMPTS, search_prompts, get_random_prompts


def append_bot_response(chat_history, response):
    """Append a bot response to Gradio 3.x pair-format chat history.

    The tool-use bot returns {"reply": str, "images": list[str]}: the reply
    fills the pending assistant slot, then each image gets its own history row
    (Gradio renders one file per message). Plain strings render as-is.
    """
    if isinstance(response, dict) and "reply" in response:
        chat_history[-1][1] = response.get("reply") or ""
        for path in response.get("images") or []:
            chat_history.append([None, (path,)])
    # Defensive: process_single_input always returns the dict above; these
    # branches keep plain-string/legacy responses renderable (unit-tested).
    elif isinstance(response, str):
        chat_history[-1][1] = response
    else:
        chat_history[-1][1] = str(response)
    return chat_history


def create_chat_interface():
    """Create and return the Gradio chat interface using the tool use pattern."""
    # Build the heavy, conversation-stateless components ONCE. Each browser
    # session gets its own isolated chatbot (fresh context + token counter) via
    # new_session(), held in gr.State so users never share conversation state.
    base_bot = SeismicChatBotToolUse()

    def respond(message, chat_history, session_bot):
        """Process a user message using a per-session chatbot (isolated context)."""
        if session_bot is None:
            session_bot = base_bot.new_session()

        chat_history = chat_history or []
        chat_history.append([message, None])
        try:
            response = session_bot.process_single_input(message)
            chat_history = append_bot_response(chat_history, response)

            # Per-session token usage for display
            token_usage = session_bot.context_manager.get_token_usage()
            token_str = f"Prompt: {token_usage['prompt_tokens']} | Completion: {token_usage['completion_tokens']} | Total: {token_usage['total_tokens']}"
            return "", chat_history, token_str, session_bot

        except Exception as e:
            chat_history[-1][1] = f"Error processing request: {str(e)}"
            return "", chat_history, "", session_bot
    
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

        **💡 Tip:** Use the example prompts below to get started quickly!
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                chat_display = gr.Chatbot(height=600)
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
        
        submit.click(respond, [msg, chat_display, session_state], [msg, chat_display, token_usage_display, session_state])
        msg.submit(respond, [msg, chat_display, session_state], [msg, chat_display, token_usage_display, session_state])
    
    return demo

if __name__ == "__main__":
    demo = create_chat_interface()
    demo.launch()
