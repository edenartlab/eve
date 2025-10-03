import os
import datetime
import random
from anthropic import Anthropic
from typing import Generator, List, Tuple

def split_conversations(text: str) -> List[str]:
    """Split the entire text into individual conversation threads.
    
    Args:
        text (str): The complete input text containing multiple conversations
        
    Returns:
        List[str]: List of individual conversation threads
    """
    # Split on the boundary marker
    conversations = text.split("=====\nTitle:")
    # Remove any empty strings and add the title marker back
    conversations = ["=====\nTitle:" + conv for conv in conversations[1:] if conv.strip()]
    return conversations

def create_chunk_from_threads(available_threads: List[str], target_size: int) -> Tuple[str, List[str]]:
    """Create a chunk by sampling threads until reaching target size.
    
    Args:
        available_threads (List[str]): List of threads that haven't been used yet
        target_size (int): Target size for the chunk in characters
        
    Returns:
        Tuple[str, List[str]]: 
            - The created chunk
            - Updated list of available threads
    """
    chunk = ""
    used_threads = []
    
    while available_threads and len(chunk) < target_size:
        # Randomly select a thread
        thread = random.choice(available_threads)
        
        # Add it to the chunk
        chunk += thread
        
        # Track which thread we used and remove it from available threads
        used_threads.append(thread)
        available_threads.remove(thread)
    
    return chunk, available_threads

def chunk_conversations(text: str, chunk_size: int) -> Generator[Tuple[str, int], None, None]:
    """Split text into chunks by randomly sampling conversation threads.
    
    Args:
        text (str): The input text to be chunked
        chunk_size (int): Target size for each chunk in characters
        
    Yields:
        Generator[Tuple[str, int], None, None]: Tuple containing:
            - chunk of text
            - number of threads remaining
    """
    # Split into individual threads
    print("Splitting conversations...")
    threads = split_conversations(text)
    print(f"Found {len(threads)} total agent/user threads.")
    
    # Shuffle the threads
    random.shuffle(threads)
    
    # Keep creating chunks until we run out of threads
    available_threads = threads.copy()
    print("Generating chunks...")
    while available_threads:
        chunk, available_threads = create_chunk_from_threads(available_threads, chunk_size)
        yield chunk, len(available_threads)

def setup_output_directory() -> str:
    """Create a timestamped directory for analysis outputs.
    
    Returns:
        str: Path to the created directory
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("analysis_" + timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def process_file(filepath: str, chunk_size: int) -> None:
    """Process a text file in chunks and send to Claude API.
    
    Args:
        filepath (str): Path to the text file to process
        chunk_size (int): Size of each chunk in characters
        
    Raises:
        ValueError: If ANTHROPIC_API_KEY is not found in environment variables
    """
    # Initialize Anthropic client
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    
    client = Anthropic(api_key=api_key)

    prompt_start = """
You are analyzing conversations between users and an AI creative assistant platform that offers a wide range of AI tools for image creation/editing, video generation, audio generation, and video composition. The goal is to identify specific, actionable use cases that can inform targeted marketing campaigns and product development.

For each conversation, carefully analyze:
1. What is the user trying to achieve? (their end goal)
2. What is their professional context or industry? (if apparent)
3. What specific problem or pain point are they trying to solve?
4. What value does the AI assistant provide in solving this problem?

As you read through these conversations, look for patterns that could indicate distinct market segments and use cases.

Here are the conversations to analyze:
--------------

"""

    prompt_end = """
--------------

Based on your analysis of these conversations, complete the following tasks:

1. Identify 10-15 specific use cases that represent distinct ways users are leveraging the creative AI assistant. For each use case, provide:
   - A clear title that describes the use case (e.g., "Rapid Social Media Content Creation for Small Businesses")
   - The user's job role or industry context
   - The specific problem or pain point being solved
   - The value proposition of using AI for this purpose
   - Estimated percentage of conversations that fit this use case

2. For each use case, provide a brief example of how this insight could be used in marketing:
   - Target audience description
   - Key message or value proposition
   - Specific pain points to address
   - Suggested marketing channels or approaches

Only include use cases that appear multiple times in the conversations and represent clear, actionable market opportunities. Focus on patterns that reveal specific professional or creative needs being addressed by the AI assistant.

Format your response in a clear, structured way that can be easily parsed for the second phase of analysis."""

    try:
        # Create output directory
        output_dir = setup_output_directory()
        
        # Read the entire file
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Estimate total chunks
        total_chunks = max(1, len(text) // chunk_size)
        print(f"Loaded {len(text)} characters from file. Chunking into aproximately {total_chunks} chunks of {chunk_size} characters.")
        
        # Process text in chunks
        for i, (chunk, remaining_threads) in enumerate(chunk_conversations(text, chunk_size), 1):
            print(f"\nProcessing chunk {i} of {total_chunks} (estimated)...")
            print(f"Remaining threads: {remaining_threads}")
            
            # Save the chunk text
            chunk_text_file = os.path.join(output_dir, f"chunk_{i}_text.txt")
            with open(chunk_text_file, 'w', encoding='utf-8') as f:
                f.write(chunk)
            print(f"Chunk text saved to: {chunk_text_file}")
            
            # Create message with the chunk
            response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=4048,
                messages=[{
                    "role": "user",
                    "content": prompt_start + chunk + prompt_end
                }]
            )
            
            # Get the response text
            response_text = ". ".join(
                [r.text for r in response.content if r.type == "text" and r.text]
            )
            
            # Save response to file
            result_file = os.path.join(output_dir, f"chunk_{i}_result.txt")
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write(response_text)
            
            print(f"Analysis result saved to: {result_file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process text file with Claude API')
    parser.add_argument('filepath', help='Path to the text file')
    parser.add_argument('--chunk-size', type=int, default=50000,
                      help='Size of each chunk in characters (default: 50000)')
    args = parser.parse_args()
    
    process_file(args.filepath, args.chunk_size)