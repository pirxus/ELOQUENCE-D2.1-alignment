# This program is to take have the punctuation to the line or the file
### "process_line" method processes the line.
### "process_file" method processed the whole file. 

from typing import List
import re
from punctuators.models import PunctCapSegModelONNX

class TextProcessor:
    def __init__(self, model_name: str):
        # Instantiate the punctuation model
        self.model = PunctCapSegModelONNX.from_pretrained(model_name)
    def add_punctuation(self, text: str) -> str:
        input_text = [text]
        out_sentence = self.model.infer(input_text)
        return self.extract_string(out_sentence)
    @classmethod
    def clean_text(self, text: str) -> str:
        cleaned_text = re.sub(r'<[^>]*>', '', text)
        cleaned_text = re.sub(r'\[.*?\]', '', cleaned_text)
        # cleaned_text = cleaned_text.replace(".Speaker:", ". Speaker")
        # cleaned_text = cleaned_text.replace("?Speaker:", "? Speaker")
        # Replace multiple spaces and incorrect punctuations
        cleaned_text = cleaned_text.replace("  ", " ")
        cleaned_text = cleaned_text.replace("?.", "?")
        cleaned_text = cleaned_text.replace(",.", ".")
        cleaned_text = cleaned_text.replace(",?", "?")
        cleaned_text = cleaned_text.replace(". .", ".")
        cleaned_text = cleaned_text.replace("? ?", "?")
        cleaned_text = cleaned_text.replace(". ,", ".")
        cleaned_text = cleaned_text.replace("-", "")
        # Correct punctuation and contractions
        cleaned_text = re.sub(r'\b(?:im|i m)\b', "I'm", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'\b(?:P\.?\s?M\.?|P\s?M|P\.?\s?m|p\.?\s?m)\b', 'pm', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = cleaned_text.replace("Youre", "You're")
        cleaned_text = cleaned_text.replace("youre", "you're")
        cleaned_text = cleaned_text.replace("You. re", "You're")
        cleaned_text = cleaned_text.replace("you. re", "you're")
        cleaned_text = re.sub(r'([.?,])\1+', r'\1', cleaned_text)
        cleaned_text = cleaned_text.replace(" ?", "?")
        cleaned_text = cleaned_text.replace(" .", ".")
        cleaned_text = cleaned_text.replace(" ,", ",")
        cleaned_text = cleaned_text.replace("?", "? ")
        cleaned_text = cleaned_text.replace(".", ". ")
        cleaned_text = cleaned_text.replace(",", ", ")
        cleaned_text = cleaned_text.replace("!", "! ")
        cleaned_text = cleaned_text.replace("\t", " ")
        cleaned_text = cleaned_text.replace("  ", " ")
        cleaned_text = cleaned_text.replace("  ", " ")
        # Handle punctuation sequences
        cleaned_text = re.sub(r'\.\s*,\s*,', '. ', cleaned_text)  # Replace ". , ," with ". "
        cleaned_text = re.sub(r'\.\s*,', '. ', cleaned_text)      # Replace ". ," with ". "
        cleaned_text = re.sub(r',\s*\.', '. ', cleaned_text)      # Replace ", ." with ". "
        cleaned_text = re.sub(r'\.\s*\?', '? ', cleaned_text)      # Replace ". ?" with "? "
        cleaned_text = re.sub(r',\s*\?', '? ', cleaned_text)      # Replace ", ?" with "? "
        # Replace multiple consecutive occurrences of the same punctuation mark with a single occurrence
        cleaned_text = re.sub(r'([.?,])\1+', r'\1', cleaned_text)
        cleaned_text = re.sub(r'([.?,])\s*([.?,])', r'\1 \2', cleaned_text)  # Ensures proper spacing
        cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)  # Replace multiple spaces with a single space
        cleaned_text = cleaned_text.replace('(', '')
        cleaned_text = cleaned_text.replace(')', '')
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = cleaned_text.strip()
        return cleaned_text

    def process_line(self, line: str) -> str:
        # Add punctuation
        punctuated_text = self.add_punctuation(line)
        # Clean the text
        cleaned_text = self.clean_text(punctuated_text)
        return cleaned_text

    def process_file(self, input_file: str, output_file: str):
        with open(input_file, 'r') as in_file, open(output_file, 'a') as out_file:
            for line in in_file:
                line = line.strip()
                # Process each line
                processed_line = self.process_line(line)
                # Write the processed line to the output file
                out_file.write(processed_line + '\n')

    def extract_string(self, nested_list):
        result = ''
        def traverse_list(lst):
            nonlocal result
            for item in lst:
                if isinstance(item, list):
                    traverse_list(item)
                elif isinstance(item, str):
                    result += item
        traverse_list(nested_list)
        return result

# Example usage
if __name__ == "__main__":
    # Initialize the processor with the model name
    processor = TextProcessor("pcs_en")

    # Specify input and output file paths
    input_file_path = '/mnt/matylda4/hegde/olmo/spotify/0_50_samples'
    output_file_path = '/mnt/matylda4/hegde/olmo/spotify_sample_punct'

    # Process the file
    processor.process_file(input_file_path, output_file_path)
