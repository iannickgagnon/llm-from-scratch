
# External imports
import tiktoken

# Context tokens
TOKEN_ENDOFTEXT = '<|endoftext|>'


if __name__ == '__main__':

    # Initialize the BPE tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Example text
    text = "Hello, world! This is a test.<|endoftext|>"

    # Encode the text
    encoded_text = tokenizer.encode(text, allowed_special={TOKEN_ENDOFTEXT})
    print(f"Encoded text: {encoded_text}")

    # Decode the text
    decoded_text = tokenizer.decode(encoded_text)
    print(f"Decoded text: {decoded_text}")