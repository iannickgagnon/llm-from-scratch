
# External libraries
import os
import re
import urllib.request


# Internal constants
URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
FILENAME = "the_verdict.txt"

# Context tokens
TOKEN_UNKNOWN = '<|unk|>'
TOKEN_ENDOFTEXT = '<|endoftext|>'


def load_text_file(url: str, filename: str, preview_length: int = 100) -> str:
    """
    Downloads a text file from a given URL and saves it locally if it does not already exist.

    Args:
        url (str): The URL of the text file to download.
        filename (str): The local file path where the downloaded file will be saved.
        preview_length (int): The number of characters to preview from the downloaded text. Default is 100.

    Returns:
        text (str): The content of the downloaded text file.

    Raises:
        urllib.error.HTTPError: If an HTTP error occurs during the download.
        urllib.error.URLError: If a URL-related error occurs.
        urllib.error.ContentTooShortError: If the download is incomplete.
        IOError: If an error occurs while writing the file.
        Exception: For any other unexpected errors.
    """
    try:

        if not os.path.exists(filename):
            
            # Download the file and save it locally
            with urllib.request.urlopen(url) as response:
                data = response.read()
                with open(filename, 'wb') as f:
                    f.write(data)
            
            # Printout
            print(f"Downloaded successfully to '{filename}'")
            print(f"Total number of characters: {len(data)}")
            print(data.decode('utf-8')[:preview_length])

        else:
            print(f"File already exists: '{filename}'")

        # Read the file and return it
        with open(filename, 'r') as f:
            return f.read()
        
    except urllib.error.HTTPError as e:
        print(f"HTTP error: {e.code} - {e.reason}")
    except urllib.error.URLError as e:
        print(f"URL error: {e.reason}")
    except urllib.error.ContentTooShortError as e:
        print("Download was incomplete.")
    except IOError as e:
        print(f"File write error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")



def tokenize_text(text: str, preview_length: int = 50, verbose: bool = False) -> list[str]:
    """
    Tokenizes a given string.

    Args:
        text (str): The text to tokenize.
        preview_length (int): The number of characters to preview from the tokenized text. Default is 50.
        verbose (bool): If True, prints the number of tokens and the first few tokens. Default is False.
        
    Returns:
        list: A list of tokens (words).
    """
    
    '''
    To split on punctuation and whitespaces, we use the following 
    regular expression : [,:;?_!"()\']|--|\s

        1. [,:;?_!"()\'] matches any of the characters inside the brackets (i.e. punctuation)
        2. | is the logical OR operator
        3. -- matches the double hyphen (used for dashes)
        4. \s matches any whitespace character (space, tab, newline)
    '''

    # Tokenize
    '''
    tokens_with_spaces = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    tokens_without_spaces = [tok.strip() for tok in tokens_with_spaces if tok.strip()]
    '''

    # This method removes the need for filtering
    tokens_without_spaces = re.findall(r"\b\w+\b|[,:;?_!\"()\']|--", text)
    tokens_without_spaces.extend((TOKEN_ENDOFTEXT, TOKEN_UNKNOWN))

    # Printout
    if verbose:
        print(f"Number of tokens: {len(tokens_without_spaces)}")
        print(f"First {preview_length} tokens: {tokens_without_spaces[:preview_length]}")

    return tokens_without_spaces


def create_vocabulary_encoder(tokens: list[str], preview_length: int = 50) -> dict[str, int]:
    """
    Creates a vocabulary from a list of tokens.

    Args:
        tokens (list): The list of tokens.
        preview_length (int): The number of tokens to preview from the vocabulary. Default is 50.

    Returns:
        dict: A dictionary mapping each unique token to its index.
    """
    
    # Create vocabulary
    vocab = {token: i for i, token in enumerate(sorted(set(tokens)))}

    # Printout
    print(f"Vocabulary size: {len(vocab)}")
    print(f"First {preview_length} tokens in vocabulary:")
    for i, (token, index) in enumerate(vocab.items()):
            if i == preview_length:
                break
            print(f"({token}, {index})")
    
    return vocab


def create_vocabulary_decoder(vocabulary: dict[str, int]) -> dict[int, str]:
    """
    Creates a decoder from a vocabulary.

    Args:
        vocabulary (dict): The vocabulary mapping each unique token to its index.

    Returns:
        dict: A dictionary mapping each index back to its token.
    """
    
    # Create decoder
    decoder = {index: token for token, index in vocabulary.items()}
    
    return decoder


def encode_text(text: str, encoder: dict[str, int]) -> list[int]:
    """
    Encodes a given text into a list of integers using a provided encoder.

    Args:
        text (str): The input text to be encoded.
        encoder (dict): A dictionary mapping tokens to their corresponding integer encodings.

    Returns:
        list[int]: A list of integers representing the encoded text. Only tokens present in the encoder are included.

    Raises:
        KeyError: If a token in the text is not found in the encoder.
    """
    return [encoder[token] if token in encoder else encoder[TOKEN_UNKNOWN] for token in tokenize_text(text)]


def decode_tokens(token_ids: list[int], decoder: dict[int, str]) -> str:
    """
    Decodes a list of token IDs into a string using a provided decoder dictionary.

    Args:
        token_ids (list[int]): A list of integer token IDs to decode.
        decoder (dict): A dictionary mapping token IDs (int) to their corresponding string representations.

    Returns:
        str: The decoded string formed by joining the mapped tokens with spaces.
    """
    return " ".join(decoder[idx] for idx in token_ids)


if __name__ == "__main__":
    
    try:

        # Load and read the text
        text = load_text_file(URL, FILENAME)

        # Tokenize
        tokens = tokenize_text(text, verbose=True)

        # Create vocabulary encoder and decoder
        encoder = create_vocabulary_encoder(tokens)
        decoder = create_vocabulary_decoder(encoder)

        # Test: encode and decode a string
        test_string = "At Be Begin Burlington"
        encoded = encode_text(test_string, encoder)
        decoded = decode_tokens(encoded, decoder)

        print(f"Original string: {test_string}")
        print(f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")

    except RuntimeError as e:
        print(f"Error occurred: {e}")
