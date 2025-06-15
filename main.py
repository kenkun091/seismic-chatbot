import logging
from config.settings import LOG_LEVEL, LOG_FORMAT
from interfaces.gradio_interface import create_chat_interface

# Create the demo interface at module level for hot reloading
demo = create_chat_interface()

def main():
    """Main entry point for the seismic chatbot application."""
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Launch the chat interface
        logger.info("Starting Seismic Modeling Assistant...")
        demo.launch(share=True)
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        raise

if __name__ == "__main__":
    main()
