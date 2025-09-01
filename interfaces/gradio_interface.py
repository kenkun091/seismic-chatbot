import gradio as gr
from core.chatbot_tool_use import SeismicChatBotToolUse
from config.example_prompts import EXAMPLE_PROMPTS, search_prompts, get_random_prompts

def create_chat_interface():
    """Create and return the Gradio chat interface using the tool use pattern."""
    seismic_bot = SeismicChatBotToolUse()
    
    # Reset token usage when interface is created (browser refresh)
    seismic_bot.context_manager.clear_context()
    
    def respond(message, chat_history):
        """Process user message and generate response using tool use pattern."""
        try:
            response = seismic_bot.process_single_input(message)
            
            # Convert to Gradio 3.x compatible format
            chat_history.append([message, None])
            
            # Handle different response types
            if isinstance(response, dict) and 'image_path' in response:
                # Handle image response
                chat_history[-1][1] = (response['image_path'],)
            elif isinstance(response, str):
                # Handle text response
                chat_history[-1][1] = response
            else:
                # Handle other response types
                chat_history[-1][1] = str(response)
                
            # Get token usage for display
            token_usage = seismic_bot.context_manager.get_token_usage()
            return "", chat_history, f"Prompt: {token_usage['prompt_tokens']} | Completion: {token_usage['completion_tokens']} | Total: {token_usage['total_tokens']}"  
                
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            chat_history[-1][1] = error_msg
            
            # Return empty token usage on error
            return "", chat_history, ""
    
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
        gr.Markdown("""
        # 🌊 Seismic Modeling Assistant (Tool Use Pattern)
        
        Welcome to the Seismic Modeling Assistant! I can help you with:
        
        - Creating and analyzing Ricker wavelets
        - Building wedge models for seismic analysis
        - Calculating AVO reflectivity using Zoeppritz and Shuey equations
        - Answering questions about seismic properties
        - Explaining seismic modeling concepts
        
        # **Available Tools:**
        # - `make_ricker`: Create Ricker wavelets with specified frequency
        # - `plot_ricker`: Plot wavelets with time and frequency analysis
        # - `wedge_model`: Create wedge models for seismic analysis
        # - `plot_wedge_model`: Plot wedge model results
        # - `zoeppritz_reflectivity`: Calculate reflectivity using Zoeppritz equations
        # - `shuey_reflectivity`: Calculate reflectivity using Shuey's approximation
        # - `plot_avo_reflectivity`: Plot AVO reflectivity curves
        
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
        
        submit.click(respond, [msg, chat_display], [msg, chat_display, token_usage_display])
        msg.submit(respond, [msg, chat_display], [msg, chat_display, token_usage_display])
    
    return demo

if __name__ == "__main__":
    demo = create_chat_interface()
    demo.launch()
