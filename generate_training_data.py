# 1_generate_data_masked_augmented_large.py
# This script uses Masked Language Modeling and Augmentation to create a massive,
# superior dataset for training a true end-to-end model.

import random
import collections
import pandas as pd
from tqdm import tqdm
import multiprocessing
import string

# --- Worker Initialization and Function ---

# Global variables for the worker processes
full_dictionary_list = None
alphabet = string.ascii_lowercase

def init_worker(dictionary_file):
    """Initializes each worker process by loading the dictionary."""
    global full_dictionary_list
    with open(dictionary_file, "r") as f:
        full_dictionary_list = f.read().splitlines()

def generate_masked_data_for_word(_):
    """
    Creates multiple training examples from a single word using masking and augmentation.
    """
    global full_dictionary_list, alphabet
    
    # 1. Select a word
    original_word = random.choice(full_dictionary_list)
    
    # 2. Data Augmentation (50% chance to slightly mutate the word)
    if random.random() < 0.5 and len(original_word) > 2:
        idx_to_change = random.randint(0, len(original_word) - 1)
        new_char = random.choice(alphabet)
        mutated_word_list = list(original_word)
        mutated_word_list[idx_to_change] = new_char
        word_to_process = "".join(mutated_word_list)
    else:
        word_to_process = original_word

    # 3. Masked Language Modeling
    # Create several masked versions of the word
    generated_samples = []
    word_len = len(word_to_process)
    if word_len < 2:
        return []

    # --- MODIFICATION: Generate many more samples per word ---
    num_samples_per_word = 20 # Increased from ~6 to 20

    # Add the fully blank version once
    generated_samples.append({
        'pattern': "_" * word_len,
        'target_word': word_to_process
    })

    # Generate many more randomly masked versions
    for _ in range(num_samples_per_word - 1): # -1 because we already added the blank one
        # Ensure we don't try to mask more letters than exist
        if word_len > 1:
            num_to_mask = random.randint(1, word_len - 1)
            word_as_list = list(word_to_process)
            indices_to_mask = random.sample(range(word_len), num_to_mask)
            
            masked_word_list = list(word_as_list)
            for idx in indices_to_mask:
                masked_word_list[idx] = '_'
            
            masked_word = "".join(masked_word_list)
            
            generated_samples.append({
                'pattern': masked_word,
                'target_word': word_to_process
            })
        
    return generated_samples

if __name__ == "__main__":
    DICTIONARY_FILE = "words_250000_train.txt"
    # We simulate "words" not "games". Let's aim for a very large dataset.
    NUM_WORDS_TO_PROCESS = 1000000 # Significantly increased for a massive dataset
    
    try:
        num_processes = multiprocessing.cpu_count()
    except NotImplementedError:
        num_processes = 4 
    
    print(f"Starting data generation with {num_processes} processes...")
    pool = multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=(DICTIONARY_FILE,))
    
    chunksize = 500 # Increased chunksize for better performance with more tasks
    results = []
    with tqdm(total=NUM_WORDS_TO_PROCESS) as pbar:
        for result_chunk in pool.imap_unordered(generate_masked_data_for_word, range(NUM_WORDS_TO_PROCESS), chunksize=chunksize):
            results.extend(result_chunk)
            pbar.update(1)
            
    pool.close()
    pool.join()

    print("\nSimulations complete. Combining results and saving to CSV...")
    df = pd.DataFrame(results)
    # The new CSV has a different structure
    df.to_csv("hangman_masked_training_data.csv", index=False)
    print(f"Successfully generated and saved {len(df)} training examples to hangman_masked_training_data.csv")
