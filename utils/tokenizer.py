import torch
import re


class TextTokenizer:
    """
    Simple tokenizer for geometric shapes and characters dataset.
    """
    def __init__(self, max_length=12):
        self.max_length = max_length
        
        # Define vocabulary
        # Special tokens
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        
        # Common words in our dataset
        self.vocab = [
            self.pad_token, self.unk_token,
            "a", "the", "in", "small", "medium", "large",
            "circle", "square", "triangle", "rectangle", "ellipse",
            "centered", "top", "bottom", "left", "right",
            "letter", "font", "plain", "bold", "italic"
        ]
        
        # Add letters A-Z
        for c in "abcdefghijklmnopqrstuvwxyz":
            self.vocab.append(c)
        
        # Create token to ID mapping
        self.token2id = {token: i for i, token in enumerate(self.vocab)}
        self.id2token = {i: token for i, token in enumerate(self.vocab)}
        
        # Set pad token ID
        self.pad_token_id = self.token2id[self.pad_token]
        self.unk_token_id = self.token2id[self.unk_token]
        
    def tokenize(self, text):
        """
        Split text into tokens.
        """
        # Convert to lowercase and split by whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def encode(self, text, return_tensors="pt"):
        """
        Encode text into token IDs.
        
        Args:
            text: String or list of strings to encode
            return_tensors: Return format ('pt' for PyTorch tensors)
        
        Returns:
            Token IDs tensor of shape [batch_size, seq_len]
        """
        # Handle single string vs. list of strings
        if isinstance(text, str):
            text = [text]
        
        batch_ids = []
        for t in text:
            # Tokenize
            tokens = self.tokenize(t)
            
            # Convert to IDs
            ids = [self.token2id.get(token, self.unk_token_id) for token in tokens]
            
            # Truncate or pad
            if len(ids) > self.max_length:
                ids = ids[:self.max_length]
            else:
                ids = ids + [self.pad_token_id] * (self.max_length - len(ids))
            
            batch_ids.append(ids)
        
        # Convert to tensor if requested
        if return_tensors == "pt":
            return torch.tensor(batch_ids)
        
        return batch_ids
    
    def decode(self, token_ids):
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Tensor of shape [batch_size, seq_len]
        
        Returns:
            List of decoded strings
        """
        # Convert to list if it's a tensor
        if torch.is_tensor(token_ids):
            token_ids = token_ids.cpu().numpy().tolist()
        
        decoded = []
        for ids in token_ids:
            # Convert IDs to tokens
            tokens = [self.id2token.get(id, self.unk_token) for id in ids]
            
            # Remove padding tokens
            tokens = [token for token in tokens if token != self.pad_token]
            
            # Join tokens
            text = " ".join(tokens)
            decoded.append(text)
        
        return decoded
    
    def get_vocab_size(self):
        """
        Get the vocabulary size of the tokenizer.
        
        Returns:
            Integer representing the vocabulary size
        """
        return len(self.vocab)
