# External imports
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

# Context tokens
TOKEN_ENDOFTEXT = '<|endoftext|>'


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        """
        Initializes the dataset by tokenizing the input text and creating overlapping 
        sequences of input and target token IDs using a sliding window approach.
        
        Args:
            txt (str): The text to be tokenized and processed.
            tokenizer (Tokenizer): The tokenizer object used to encode the text.
            max_length (int): The maximum length of each sequence|chunk.
            stride (int): The step size for the sliding window.
        
        Attributes:
            input_ids (list of torch.Tensor): A list of tensors containing input token ID sequences.
            target_ids (list of torch.Tensor): A list of tensors containing target token ID sequences, 
                which are offset by one position from the input sequences.
        """

        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={TOKEN_ENDOFTEXT})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1: i + max_length + 1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt: str,
                         batch_size: int,
                         max_length: int,
                         stride: int,
                         shuffle: bool = False,
                         drop_last: bool = True,
                         num_workers: int = 0) -> tuple:
    """
    Creates a PyTorch DataLoader for tokenized text data using a custom GPTDatasetV1.
    
    Args:
        txt (str): The input text to be tokenized and used for creating the dataset.
        batch_size (int): The number of samples per batch to load.
        max_length (int): The maximum sequence length for tokenized text.
        stride (int): The stride size for creating overlapping sequences.
        shuffle (bool, optional): Whether to shuffle the data at every epoch. Defaults to False.
        drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to True.
        num_workers (int, optional): The number of subprocesses to use for data loading. Defaults to 0.
    
    Returns:
        tuple: A tuple containing:
            - dataloader (DataLoader): The PyTorch DataLoader for the dataset.
            - tokenizer (tiktoken.Encoding): The tokenizer used for encoding the text.
    """
    
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last, 
        num_workers=num_workers
        )

    return dataloader, tokenizer


if __name__ == '__main__':

    # Load the text file
    with open("the_verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    vocab_size = 50257      # Size of the vocabulary (GPT-2 tokenizer)
    output_dim = 256        # Size of the output dimension for the token and positional embeddings
    context_length = 1024   # Maximum context length for the model

    # Initialize the token and positional embedding layers with random weights
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    batch_size = 8  # Number of sequences in each batch
    max_length = 4  # Maximum length of each sequence|chunk

    # Create the dataloader and tokenizer
    dataloader, tokenizer = create_dataloader_v1(
        raw_text,
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length
    )

    for batch in dataloader:
        
        # Unpack th current batch
        x, y = batch

        print(f"Input = {x}")
        print()

        print(f"Target = {y}")
        print()

        print(f"Input shape = {x.shape}")
        print(f"Target shape = {y.shape}")
        print()

        # Decode the first input-target pair
        print("First input-target pair:")
        print(f"\tDecoded input = {tokenizer.decode(x[0].tolist())}")
        print(f"\tDecoded target = {tokenizer.decode(y[0].tolist())}")
        print()

        print("Second input-target pair:")
        print(f"\tDecoded input = {tokenizer.decode(x[1].tolist())}")
        print(f"\tDecoded target = {tokenizer.decode(y[1].tolist())}")
        print()

        # Get the input and target sequences
        token_embeddings = token_embedding_layer(x)
        pos_embeddings = pos_embedding_layer(torch.arange(max_length))

        # Add positional information to the token embeddings
        input_embeddings = token_embeddings + pos_embeddings

        break

    print(input_embeddings.shape)