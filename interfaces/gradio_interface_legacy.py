import gradio as gr
from core.chatbot import SeismicChatBot

def create_chat_interface():
    """Create and return the Gradio chat interface using the legacy implementation."""
    seismic_bot = SeismicChatBot()
    
    def respond(message, chat_history):
        """Process user message and generate response using legacy pattern."""
        try:
            response = seismic_bot.process_input(message)
            
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
    
    with gr.Blocks(title="Seismic Modeling Assistant - Legacy") as demo:
        gr.Markdown("""
        # 🌊 Seismic Modeling Assistant (Legacy Implementation)
        
        Welcome to the Seismic Modeling Assistant! I can help you with:
        
        - Creating and analyzing Ricker wavelets
        - Building wedge models for seismic analysis
        - Answering questions about seismic properties
        - Explaining seismic modeling concepts
        
        **Note:** This is the legacy implementation. For the new tool use pattern, 
        use the default interface.
        
        Try asking me something like:
        - "What is a Ricker wavelet?"
        - "Create a 30 Hz Ricker wavelet"
        - "Explain tuning effects in wedge models"
        - "Make a wedge model with 100m thickness"
        """)
        
        with gr.Row():
            with gr.Column(scale=4):
                chat_display = gr.Chatbot(height=600, type='messages')
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask a question or request an action...",
                        show_label=False,
                        container=False
                    )
                    submit = gr.Button("Send", variant="primary")
            
            with gr.Column(scale=1):
                gr.Markdown("""
                ### Quick Examples
                
                **Questions:**
                - What is a Ricker wavelet?
                - How does frequency affect resolution?
                - Explain tuning thickness
                
                **Actions:**
                - Create 30 Hz Ricker
                - Plot wavelet spectrum
                - Make wedge model
                """)
        
        submit.click(respond, [msg, chat_display], [msg, chat_display])
        msg.submit(respond, [msg, chat_display], [msg, chat_display])
    
    return demo

if __name__ == "__main__":
    demo = create_chat_interface()
    demo.launch() 