import gradio as gr
from core.chatbot import SeismicChatBot

def create_chat_interface():
    """Create and return the Gradio chat interface."""
    seismic_bot = SeismicChatBot()
    
    def respond(message, chat_history):
        """Process user message and generate response."""
        response = seismic_bot.process_input(message)
        # Convert to new message format
        chat_history.append({"role": "user", "content": message})
        # Handle image response
        if isinstance(response, dict) and 'image_path' in response:
            # Gradio expects image as a tuple: (image_path,)
            chat_history.append({"role": "assistant", "content": (response['image_path'],)})
        else:
            chat_history.append({"role": "assistant", "content": response})
        return "", chat_history
    
    with gr.Blocks(title="Seismic Modeling Assistant") as demo:
        gr.Markdown("""
        # 🌊 Seismic Modeling Assistant
        
        Welcome to the Seismic Modeling Assistant! I can help you with:
        
        - Creating and analyzing Ricker wavelets
        - Building wedge models for seismic analysis
        - Answering questions about seismic properties
        - Explaining seismic modeling concepts
        
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
