import random
import collections
import pandas as pd
from tqdm import tqdm
import multiprocessing
import string

def generate_masked_data_for_word(word_to_process):
    """
    Creates 80 random, game-accurate training examples for a single word.
    """
    generated_samples = []
    word_len = len(word_to_process)
    if word_len < 2:
        return []

    num_samples_per_word = 80
    unique_letters = list(set(word_to_process))

    for _ in range(num_samples_per_word):
        # --- RANDOM VALID PATTERN LOGIC ---
        
        # 1. Decide how many unique letters to reveal (from 0 to all but one)
        # This ensures the word is never fully revealed in the pattern.
        if len(unique_letters) > 1:
            num_letters_to_reveal = random.randint(0, len(unique_letters) - 1)
        else:
            num_letters_to_reveal = 0

        # 2. Randomly choose which letters to reveal
        letters_to_reveal = set(random.sample(unique_letters, num_letters_to_reveal))

        # 3. Build the pattern. If a character is in our "reveal" set, show all instances of it.
        # Otherwise, it's an underscore. This perfectly mimics the game rules.
        pattern_list = []
        for char in word_to_process:
            if char in letters_to_reveal:
                pattern_list.append(char)
            else:
                pattern_list.append('_')
        
        masked_word = "".join(pattern_list)
        
        generated_samples.append({
            'pattern': masked_word,
            'target_word': word_to_process
        })
        
    return generated_samples

if __name__ == "__main__":
    DICTIONARY_FILE = "words_250000_train.txt"
    
    # Load the full dictionary in the main process
    with open(DICTIONARY_FILE, "r") as f:
        full_dictionary_list = f.read().splitlines()
    
    try:
        num_processes = multiprocessing.cpu_count()
    except NotImplementedError:
        num_processes = 4 
    
    print(f"Starting data generation with {num_processes} processes (Balanced Random Strategy)...")
    pool = multiprocessing.Pool(processes=num_processes)
    
    chunksize = 100 # words to send to each worker process at a time
    results = []
    
    # Use the full_dictionary_list as the iterable for the parallel processes
    with tqdm(total=len(full_dictionary_list)) as pbar:
        for result_chunk in pool.imap_unordered(generate_masked_data_for_word, full_dictionary_list, chunksize=chunksize):
            results.extend(result_chunk)
            pbar.update(1)
            
    pool.close()
    pool.join()

    print("\nSimulations complete. Combining results and saving to CSV...")
    df = pd.DataFrame(results)
    df.to_csv("hangman_masked_training_data.csv", index=False)
    print(f"Successfully generated and saved {len(df)} training examples to hangman_masked_training_data.csv")
