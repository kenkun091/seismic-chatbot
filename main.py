import logging
import argparse
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
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Seismic Modeling Assistant")
    parser.add_argument(
        "--mode", 
        choices=["tool-use", "legacy"], 
        default="tool-use",
        help="Choose implementation mode: tool-use (new) or legacy (old)"
    )
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Run test examples instead of launching interface"
    )
    
    args = parser.parse_args()
    
    try:
        if args.test:
            # Run test examples
            if args.mode == "tool-use":
                logger.info("Running tool use pattern examples...")
                from example_tool_use import demonstrate_tool_use
                demonstrate_tool_use()
            else:
                logger.info("Running legacy pattern examples...")
                from test_tool_use import test_tool_use_pattern
                test_tool_use_pattern()
        else:
            # Launch the chat interface
            logger.info(f"Starting Seismic Modeling Assistant in {args.mode} mode...")
            if args.mode == "legacy":
                # Use legacy interface
                from interfaces.gradio_interface_legacy import create_chat_interface as create_legacy_interface
                demo = create_legacy_interface()
            else:
                # Use tool use interface (default)
                demo = create_chat_interface()
            
            demo.launch(share=True)
            
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        raise

if __name__ == "__main__":
    main()
