#!/usr/bin/env python3
"""
Test script to demonstrate interrupt handling and progress preservation
in the LLM extraction process.
"""

import asyncio
import time
from pathlib import Path
from agents.llm_extractor import process_csv_with_progress, get_openai_client

def progress_callback(message, current, total):
    """Simple progress callback for testing."""
    print(f"[{current}/{total}] {message}")

async def test_interrupt_handling():
    """Test the interrupt handling functionality."""
    print("🧪 Testing LLM Extractor Interrupt Handling")
    print("=" * 50)

    # Test with a sample CSV file
    csv_path = Path("crawled_data/canarymedia_com_20251128_filtered.csv")
    output_dir = Path("test_output")

    if not csv_path.exists():
        print(f"❌ Test CSV file not found: {csv_path}")
        return

    print(f"📁 Using test file: {csv_path}")
    print(f"📂 Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Initialize OpenAI client (you'll need to set your API key)
    try:
        client = get_openai_client(provider="openai")  # or "azure", "lm_studio"
        model_name = "gpt-3.5-turbo"  # or your preferred model
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI client: {e}")
        print("   Make sure your API key is set in environment variables")
        return

    print("🚀 Starting LLM extraction with checkpointing...")
    print("   (Press Ctrl+C to interrupt and test resume functionality)")

    try:
        # Start processing with checkpointing enabled
        df, stats = await process_csv_with_progress(
            csv_path=csv_path,
            output_dir=output_dir,
            client=client,
            model_name=model_name,
            text_column="text_content",
            progress_callback=progress_callback,
            checkpoint_interval=5,  # Save every 5 rows for testing
            resume_from_checkpoint=True
        )

        if stats.get('interrupted'):
            print("\n🛑 Processing was interrupted!")
            print(f"   Progress saved to: {stats.get('checkpoint_file')}")
            print(f"   Processed {stats['processed']} out of {stats['total_rows']} rows")
            print("\n🔄 To resume processing, run this script again.")
        else:
            print("\n✅ Processing completed successfully!")
            print(f"   Processed {stats['processed']} rows")
            print(f"   Output CSV: {stats.get('output_csv')}")
            print(f"   Output JSON: {stats.get('output_json')}")

    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt detected!")
        print("   Progress should have been saved automatically.")

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")

if __name__ == "__main__":
    asyncio.run(test_interrupt_handling())