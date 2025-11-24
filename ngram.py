import random
from collections import defaultdict, Counter
import re

def text_preprocessing(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep basic punctuation
    # This keeps sentence structure somewhat intact
    text = re.sub(r'[^a-z\s\.\,\!\?\;]', '', text)
    
    # Tokenize by splitting on whitespace
    tokens = text.split()
    
    # Remove empty tokens
    tokens = [token for token in tokens if token.strip()]
    
    print(f"Corpus preprocessed: {len(tokens)} tokens extracted")
    return tokens


def build_trigram_model(tokens):
    # Store trigram frequencies: (word1, word2) -> {word3: count}
    trigram_counts = defaultdict(Counter)
    
    # Store bigram frequencies for normalization: (word1, word2) -> total_count
    bigram_counts = defaultdict(int)
    
    # Build vocabulary for Laplace smoothing
    vocabulary = set(tokens)
    vocab_size = len(vocabulary)
    
    # Iterate through tokens to build trigrams
    for i in range(len(tokens) - 2):
        word1 = tokens[i]
        word2 = tokens[i + 1]
        word3 = tokens[i + 2]
        
        # Record the trigram
        trigram_counts[(word1, word2)][word3] += 1
        
        # Record the bigram prefix
        bigram_counts[(word1, word2)] += 1
    
    print(f"Trigram model built:")
    print(f"  - Vocabulary size: {vocab_size}")
    print(f"  - Unique bigram contexts: {len(bigram_counts)}")
    print(f"  - Total trigrams: {sum(bigram_counts.values())}")
    
    return trigram_counts, bigram_counts, vocab_size, vocabulary


def get_next_word_probabilities(trigram_counts, bigram_counts, vocab_size, 
                                 vocabulary, context):
    context_tuple = tuple(context)
    
    # Get count of this specific bigram context
    bigram_count = bigram_counts.get(context_tuple, 0)
    
    # Calculate probabilities with Laplace smoothing
    probabilities = {}
    
    for word in vocabulary:
        # Add-1 smoothing: add 1 to numerator, add vocab_size to denominator
        trigram_count = trigram_counts[context_tuple].get(word, 0)
        probability = (trigram_count + 1) / (bigram_count + vocab_size)
        probabilities[word] = probability
    
    return probabilities


def generate_text(trigram_counts, bigram_counts, vocab_size, vocabulary, 
                  starting_sequence, length):
    # Initialize with starting sequence
    generated = list(starting_sequence)
    
    # Generate remaining words
    for _ in range(length - len(starting_sequence)):
        # Get context (last two words)
        context = (generated[-2], generated[-1])
        
        # Get probability distribution for next word
        probabilities = get_next_word_probabilities(
            trigram_counts, bigram_counts, vocab_size, vocabulary, context
        )
        
        # Stochastic sampling: choose next word based on probabilities
        words = list(probabilities.keys())
        probs = list(probabilities.values())
        
        # Randomly select next word according to probability distribution
        next_word = random.choices(words, weights=probs, k=1)[0]
        
        generated.append(next_word)
    
    return ' '.join(generated)


def main():
    """
    Main demonstration: Load corpus, build model, generate text samples.
    """
    print("="*60)
    print("MISSION 5: N-GRAM CORE INITIALIZATION")
    print("Statistical Language Model - Trigram Implementation")
    print("="*60)
    print()
    
    # Step 1: Load and preprocess corpus
    corpus_file = "transformers.txt"  # Replace with your corpus file
    tokens = text_preprocessing(corpus_file)
    print()
    
    # Step 2: Build trigram model
    trigram_counts, bigram_counts, vocab_size, vocabulary = build_trigram_model(tokens)
    print()
    
    # Step 3: Generate text samples
    starting_sequence = ["the", "transformer"]
    num_words = 30
    num_samples = 3
    
    print("="*60)
    print(f"GENERATING {num_samples} TEXT SAMPLES")
    print(f"Starting sequence: '{' '.join(starting_sequence)}'")
    print(f"Number of words: {num_words}")
    print("="*60)
    print()
    
    with open("responses.txt", "w+") as f:
        for i in range(num_samples):
            print(f"--- SAMPLE {i+1} ---")
            generated_text = generate_text(
                trigram_counts, bigram_counts, vocab_size, vocabulary,
                starting_sequence, num_words
            )
            f.write(f"\n\nResponse {i+1}\n")
            f.write(generated_text)
            print(generated_text)
    
    print("="*60)
    print("N-GRAM CORE OPERATIONAL")
    print("="*60)




if __name__ == "__main__":
    main()