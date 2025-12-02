"""
Rebuild embeddings for all JSON files in summarised_content/
"""
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from embeddings_processor import JSONEmbeddingProcessor
import json

def rebuild_all_embeddings():
    """Rebuild embeddings for all JSON files"""
    
    # Initialize processor
    processor = JSONEmbeddingProcessor(
        enable_s3_sync=True,
        local_storage_dir="./embeddings_storage"
    )
    
    # Find all JSON files
    summarised_dir = Path("summarised_content")
    json_files = list(summarised_dir.glob("*.json"))
    
    # Filter out history.json
    json_files = [f for f in json_files if f.name != "history.json"]
    
    print(f"Found {len(json_files)} JSON files to process:")
    for f in json_files:
        print(f"  - {f.name}")
    
    print("\n" + "=" * 60)
    print("REBUILDING EMBEDDINGS")
    print("=" * 60)
    
    for json_file in json_files:
        print(f"\nProcessing: {json_file.name}")
        print("-" * 60)
        
        try:
            # Load JSON to check size
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"  Records in file: {len(data)}")
            
            # Process the file
            embedding_index = processor.process_json_file(
                json_file, # type: ignore
                progress_callback=lambda msg, current, total: print(f"  {msg}")
            )
            
            print(f"\n  ✓ Created embedding index:")
            print(f"    - Documents: {embedding_index.num_documents}")
            print(f"    - Chunks: {embedding_index.num_chunks}")
            print(f"    - Index name: {embedding_index.index_name}")
            
            # Save locally
            local_path = processor.save_index_locally(embedding_index)
            print(f"    - Saved locally: {local_path}")
            
            # Upload to S3
            if processor.upload_index_to_s3(embedding_index):
                print(f"    - ✓ Uploaded to S3")
            else:
                print(f"    - ⚠️  S3 upload failed")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print("REBUILD COMPLETE!")
    print("=" * 60)
    
    # List all indexes
    print("\nLocal embeddings:")
    local_indexes = processor.list_local_indexes()
    for idx in local_indexes:
        embedding_index = processor.load_index_locally(idx)
        print(f"  - {idx}: {embedding_index.num_documents} docs, {embedding_index.num_chunks} chunks")

if __name__ == "__main__":
    rebuild_all_embeddings()
