import gradio as gr
from core.chatbot_tool_use import SeismicChatBotToolUse
from config.example_prompts import EXAMPLE_PROMPTS, search_prompts, get_random_prompts

def create_chat_interface():
    """Create and return the Gradio chat interface using the tool use pattern."""
    seismic_bot = SeismicChatBotToolUse()
    
    def respond(message, chat_history):
        """Process user message and generate response using tool use pattern."""
        try:
            response = seismic_bot.process_single_input(message)
            
            # Convert to new message format
            chat_history.append({"role": "user", "content": message})
            
            # Handle different response types
            if isinstance(response, dict) and 'image_path' in response:
                # Handle image response
                chat_history.append({"role": "assistant", "content": (response['image_path'],)})
            elif isinstance(response, str):
                # Handle text response
                chat_history.append({"role": "assistant", "content": response})
            else:
                # Handle other response types
                chat_history.append({"role": "assistant", "content": str(response)})
                
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            chat_history.append({"role": "assistant", "content": error_msg})
            
        return "", chat_history
    
    def copy_prompt(prompt_text):
        """Copy prompt to clipboard and return it for the textbox."""
        return prompt_text
    
    def search_examples(query):
        """Search for example prompts."""
        if not query.strip():
            return get_random_prompts(5)
        return search_prompts(query)
    
    with gr.Blocks(title="Seismic Modeling Assistant - Tool Use") as demo:
        gr.Markdown("""
        # 🌊 Seismic Modeling Assistant (Tool Use Pattern)
        
        Welcome to the Seismic Modeling Assistant! I can help you with:
        
        - Creating and analyzing Ricker wavelets
        - Building wedge models for seismic analysis
        - Calculating AVO reflectivity using Zoeppritz and Shuey equations
        - Answering questions about seismic properties
        - Explaining seismic modeling concepts
        
        **Available Tools:**
        - `make_ricker`: Create Ricker wavelets with specified frequency
        - `plot_ricker`: Plot wavelets with time and frequency analysis
        - `wedge_model`: Create wedge models for seismic analysis
        - `plot_wedge_model`: Plot wedge model results
        - `zoeppritz_reflectivity`: Calculate reflectivity using Zoeppritz equations
        - `shuey_reflectivity`: Calculate reflectivity using Shuey's approximation
        - `plot_avo_reflectivity`: Plot AVO reflectivity curves
        
        **💡 Tip:** Use the example prompts below to get started quickly!
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                chat_display = gr.Chatbot(height=600, type='messages')
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask a question or request an action...",
                        show_label=False,
                        container=False
                    )
                    submit = gr.Button("Send", variant="primary")
            
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
                    
                    # Search results
                    search_results = gr.HTML(
                        value="<p><em>Search for examples or browse categories below...</em></p>",
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
        
        submit.click(respond, [msg, chat_display], [msg, chat_display])
        msg.submit(respond, [msg, chat_display], [msg, chat_display])
    
    return demo

if __name__ == "__main__":
    demo = create_chat_interface()
    demo.launch()
