"""
Test script to debug chatbot query issues
"""
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from query_tests import JSONEmbeddingProcessor

def test_embeddings_loading():
    """Test if embeddings can be loaded"""
    processor = JSONEmbeddingProcessor()
    
    print("=" * 60)
    print("TESTING EMBEDDINGS LOADING")
    print("=" * 60)
    
    # List local indexes
    local_indexes = processor.list_local_indexes()
    print(f"\nFound {len(local_indexes)} local indexes:")
    for idx in local_indexes:
        print(f"  - {idx}")
    
    # Load each index
    embedding_indexes = {}
    for index_name in local_indexes:
        try:
            embedding_index = processor.load_index_locally(index_name)
            source_name = index_name.replace('_embeddings', '')
            embedding_indexes[source_name] = embedding_index
            
            print(f"\n✓ Loaded: {index_name}")
            print(f"  Documents: {embedding_index.num_documents}")
            print(f"  Chunks: {embedding_index.num_chunks}")
            print(f"  Source: {embedding_index.source_name}")
            
        except Exception as e:
            print(f"\n✗ Failed to load {index_name}: {e}")
    
    return processor, embedding_indexes

def test_queries(processor, embedding_indexes):
    """Test various queries"""
    print("\n" + "=" * 60)
    print("TESTING QUERIES")
    print("=" * 60)
    
    test_queries = [
        "what is in your knowledge base",
        "AI technology",
        "hydrogen energy",
        "carbon markets",
        "latest developments",
        "TechCrunch",
    ]
    
    for query_text in test_queries:
        print(f"\n\nQuery: '{query_text}'")
        print("-" * 60)
        
        all_results = []
        
        for source_name, embedding_index in embedding_indexes.items():
            try:
                results = processor.query_similar(
                    embedding_index,
                    query_text=query_text,
                    top_k=3,
                    filter_metadata=None
                )
                
                print(f"\n  Source: {source_name}")
                print(f"  Results found: {len(results)}")
                
                if results:
                    for i, result in enumerate(results[:2], 1):
                        print(f"\n    Result {i}:")
                        print(f"      Score: {result['score']:.4f}")
                        print(f"      Title: {result['metadata'].get('title', 'N/A')}")
                        print(f"      Text preview: {result['text'][:100]}...")
                
                all_results.extend(results)
                
            except Exception as e:
                print(f"\n  ✗ Error querying {source_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Sort all results by score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        top_results = all_results[:5]
        
        print(f"\n  TOTAL RESULTS ACROSS ALL SOURCES: {len(all_results)}")
        scores_str = [f"{r['score']:.4f}" for r in top_results]
        print(f"  TOP 5 SCORES: {scores_str}")
        
        # Check if context would be empty
        context_texts = [r['text'] for r in top_results]
        context = "\n\n---\n\n".join(context_texts) if context_texts else "No relevant information found in the knowledge base."
        
        print(f"\n  Context length: {len(context)} chars")
        print(f"  Context preview: {context[:200]}...")

def test_sample_chunks(embedding_indexes):
    """Display sample chunks from each source"""
    print("\n" + "=" * 60)
    print("SAMPLE CHUNKS FROM EACH SOURCE")
    print("=" * 60)
    
    for source_name, embedding_index in embedding_indexes.items():
        print(f"\n\nSource: {source_name}")
        print("-" * 60)
        
        # Get first 3 chunks
        sample_chunks = embedding_index.chunks[:3]
        
        for i, chunk in enumerate(sample_chunks, 1):
            print(f"\nChunk {i}:")
            print(f"  Record ID: {chunk.record_id}")
            print(f"  Chunk ID: {chunk.chunk_id}")
            print(f"  Title: {chunk.metadata.get('title', 'N/A')}")
            print(f"  Text: {chunk.text[:150]}...")
            print(f"  Embedding dimensions: {len(chunk.embedding)}")

if __name__ == "__main__":
    # Test loading
    processor, embedding_indexes = test_embeddings_loading()
    
    if not embedding_indexes:
        print("\n❌ No embeddings loaded. Cannot continue tests.")
        exit(1)
    
    # Test sample chunks
    test_sample_chunks(embedding_indexes)
    
    # Test queries
    test_queries(processor, embedding_indexes)
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
