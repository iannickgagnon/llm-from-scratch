
# External libraries
import os
import re
import urllib.request


# Internal constants
URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
FILENAME = "the_verdict.txt"


def load_text_file(url: str, filename: str) -> str:
    """
    Downloads a text file from a given URL and saves it locally if it does not already exist.

    Args:
        url (str): The URL of the text file to download.
        filename (str): The local file path where the downloaded file will be saved.

    Returns:
        str: A message indicating the status of the operation.

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
            print(data.decode('utf-8')[:100])

        else:
            print(f"File already exists: '{filename}'")
    except urllib.error.HTTPError as e:
        print(f"HTTP error: {e.code} - {e.reason}")
    except urllib.error.URLError as e:
        print(f"URL error: {e.reason}")
    except urllib.error.ContentTooShortError as e:
        print("Download was incomplete.")
    except urllib.error.IOError as e:
        print(f"File write error: {e}")
    except urllib.error.Exception as e:
        print(f"Unexpected error: {e}")


def tokenize_text(text: str) -> list:
    """
    Tokenizes a given string.

    Args:
        text (str): The text to tokenize.

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
    tokens_with_spaces = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    tokens_without_spaces = [tok.strip() for tok in tokens_with_spaces if tok.strip()]
    
    # Printout
    print(f"Number of tokens: {len(tokens_without_spaces)}")
    print(f"First 10 tokens: {tokens_without_spaces[:50]}")

    return tokens_without_spaces


def create_vocabulary_encoder(tokens: list[str]):
    """
    Creates a vocabulary from a list of tokens.

    Args:
        tokens (list): The list of tokens.

    Returns:
        dict: A dictionary mapping each unique token to its index.
    """
    
    # Create vocabulary
    vocab = {token: i for i, token in enumerate(sorted(set(tokens)))}
    
    # Printout
    print(f"Vocabulary size: {len(vocab)}")
    print(f"First tokens in vocabulary:")
    for i, (token, index) in enumerate(vocab.items()):
            print(f"({token}, {index})")
            if i == 50:
                break
    
    return vocab


def create_vocabulary_decoder(vocabulary: dict):
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


if __name__ == "__main__":
    
    # Load text from URL
    load_text_file(URL, FILENAME)

    # Read the text file
    with open(FILENAME, 'r', encoding='utf-8') as f:
        text = f.read()

    # Tokenize the text
    tokens = tokenize_text(text)

    # Create vocabulary
    encoder = create_vocabulary_encoder(tokens)

    # Create decoder
    decoder = create_vocabulary_decoder(encoder)

    # Test encoder / decoder
    string_to_encode = "At Be Begin Burlington"
    encoded_string = [encoder[token] for token in tokenize_text(string_to_encode)]
    decoded_string = " ".join([decoder[index] for index in encoded_string])
    print(f"Original string: {string_to_encode}")
    print(f"Encoded string: {encoded_string}")
    print(f"Decoded string: {decoded_string}")
