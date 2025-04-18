
# External imports
import tiktoken

# Internal imports
from chapter_2 import (load_text_file, 
                       URL, 
                       FILENAME)

# Internal constants
CONTEXT_SIZE = 4


if __name__ == '__main__':

    # Load and read the text
    text = load_text_file(URL, FILENAME)

    # Initialize the BPE tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Encode the text (remove the 50 first tokens to reproduce the text's example)
    encoded_text = tokenizer.encode(text)[50:]

    # Show the size of the vocabulary
    print(f"Vocabulary size = {len(encoded_text)}")

    # Print sliding window of tokens
    for i in range(1, CONTEXT_SIZE + 1):
        context = encoded_text[:i]
        desired = encoded_text[i]
        print(f"{tokenizer.decode(context)} ---> {tokenizer.decode([desired])}")
