from llama_index.core.node_parser import SentenceSplitter

class SemanticProcessor:
    def __init__(self, chunk_size=512, chunk_overlap=50):
        """Initialize the semantic processor with configurable chunk parameters."""
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def process(self, text: str) -> str:
        """
        Process the input text using sentence splitting.
        
        Args:
            text: Input text to process
            
        Returns:
            Processed text with chunk markers
        """
        chunks = self.splitter.split_text(text)
        
        # Combine chunks into a single string with markers
        output = ""
        for i, chunk in enumerate(chunks):
            output += f"--- Chunk {i+1} ---\n"
            output += chunk + "\n\n"
            
        return output
