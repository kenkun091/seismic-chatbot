import os
import logging
import argparse
from config.settings import LOG_LEVEL, LOG_FORMAT
from interfaces.gradio_interface import create_chat_interface


def build_interface(mode: str):
    """Build the Gradio demo for the given --mode."""
    if mode == "legacy":
        from interfaces.gradio_interface_legacy import create_chat_interface as create_legacy
        return create_legacy()
    if mode == "agentic":
        from core.orchestrator import SeismicOrchestrator
        return create_chat_interface(base_bot=SeismicOrchestrator())
    return create_chat_interface()


def main():
    """Main entry point for the seismic chatbot application."""
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT
    )
    logger = logging.getLogger(__name__)

    # Optional OTel export of decision traces (no-op unless the OTLP endpoint
    # env vars are set; see core/otel_export.py).
    from core.otel_export import install as install_otel
    if install_otel():
        logger.info("OTel trace export enabled")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Seismic Modeling Assistant")
    parser.add_argument(
        "--mode",
        choices=["tool-use", "agentic", "legacy"],
        default="tool-use",
        help="Choose implementation mode: tool-use (new), agentic (orchestrator + subagents), or legacy (old)"
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
            demo = build_interface(args.mode)

            # Do NOT expose a public tunnel by default — that would put an
            # unauthenticated, key-billing endpoint on the internet. Opt in
            # explicitly with GRADIO_SHARE=1; otherwise bind to localhost.
            share = os.environ.get("GRADIO_SHARE", "").strip().lower() in ("1", "true", "yes")
            host = os.environ.get("GRADIO_HOST", "127.0.0.1")
            demo.launch(share=share, server_name=host)
            
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        raise

if __name__ == "__main__":
    main()
