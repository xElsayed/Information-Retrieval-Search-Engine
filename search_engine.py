"""
Information Retrieval Search Engine - Advanced Backend
Implements preprocessing, indexing, and retrieval algorithms with advanced features
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class SearchEngine:
    """Main Search Engine class implementing all IR algorithms with advanced features"""
    
    def __init__(self, corpus: List[str]):
        """
        Initialize search engine with a corpus of documents
        
        Args:
            corpus: List of document strings
        """
        self.corpus = corpus
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Build all indexes
        self.inverted_index = self._build_inverted_index()
        self.positional_index = self._build_positional_index()
        self.soundex_index = self._build_soundex_index()
        
        # Store original tokens for highlighting
        self.doc_tokens = self._build_token_map()
    
    # ==================== PREPROCESSING ====================
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words using NLTK
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        return word_tokenize(text.lower())
    
    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        """
        Remove stop words from token list
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered list without stop words
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply Porter Stemmer to tokens
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of stemmed tokens
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess(self, text: str) -> Dict[str, List[str]]:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with tokens, filtered_tokens, and stemmed_tokens
        """
        tokens = self.tokenize(text)
        filtered_tokens = self.remove_stop_words(tokens)
        stemmed_tokens = self.stem_tokens(filtered_tokens)
        
        return {
            'tokens': tokens,
            'filtered_tokens': filtered_tokens,
            'stemmed_tokens': stemmed_tokens
        }
    
    # ==================== SOUNDEX ALGORITHM ====================
    
    @staticmethod
    def soundex(name: str) -> str:
        """
        Soundex algorithm implementation following Lab 5 IR.pdf specifications
        
        Steps:
        1. Retain the first letter (uppercase)
        2. Change A, E, I, O, U, H, W, Y to '0'
        3. Map letters to digits:
           B, F, P, V → 1
           C, G, J, K, Q, S, X, Z → 2
           D, T → 3
           L → 4
           M, N → 5
           R → 6
        4. Remove pairs of consecutive duplicate digits
        5. Remove all zeros
        6. Pad with zeros and return first 4 characters (Letter + 3 digits)
        
        Args:
            name: Input name/word
            
        Returns:
            4-character Soundex code
        """
        if not name:
            return ""
        
        # Step 1: Retain the first letter (uppercase)
        name = name.upper()
        first_letter = name[0]
        working_string = name[1:]  # Rest of the string
        
        # Step 2: Change vowels and H, W, Y to '0'
        vowels_and_special = 'AEIOUHWY'
        for char in vowels_and_special:
            working_string = working_string.replace(char, '0')
        
        # Step 3: Map letters to digits according to Soundex rules
        soundex_mapping = {
            'B': '1', 'F': '1', 'P': '1', 'V': '1',
            'C': '2', 'G': '2', 'J': '2', 'K': '2', 
            'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
            'D': '3', 'T': '3',
            'L': '4',
            'M': '5', 'N': '5',
            'R': '6'
        }
        
        result_string = ""
        for char in working_string:
            if char in soundex_mapping:
                result_string += soundex_mapping[char]
            else:
                result_string += char  # Keep '0' and any other chars
        
        # Step 4: Remove pairs of consecutive duplicate digits
        deduplicated = ""
        prev_char = ""
        for char in result_string:
            if char != prev_char or char == '0':
                deduplicated += char
            prev_char = char
        
        # Step 5: Remove all zeros
        no_zeros = deduplicated.replace('0', '')
        
        # Step 6: Pad with trailing zeros and return first 4 positions
        final_code = first_letter + no_zeros + "000"
        return final_code[:4]
    
    # ==================== INDEX BUILDING ====================
    
    def _build_inverted_index(self) -> Dict[str, List[int]]:
        """
        Build inverted index: term -> [doc_ids]
        
        Returns:
            Dictionary mapping terms to list of document IDs
        """
        inverted_index = defaultdict(list)
        
        for doc_id, document in enumerate(self.corpus):
            processed = self.preprocess(document)
            stemmed_tokens = processed['stemmed_tokens']
            
            # Get unique terms in this document
            unique_terms = set(stemmed_tokens)
            
            for term in unique_terms:
                inverted_index[term].append(doc_id)
        
        return dict(inverted_index)
    
    def _build_positional_index(self) -> Dict[str, Dict[int, List[int]]]:
        """
        Build positional index: term -> {doc_id: [positions]}
        
        Returns:
            Dictionary mapping terms to documents and positions
        """
        positional_index = defaultdict(lambda: defaultdict(list))
        
        for doc_id, document in enumerate(self.corpus):
            processed = self.preprocess(document)
            stemmed_tokens = processed['stemmed_tokens']
            
            for position, term in enumerate(stemmed_tokens):
                positional_index[term][doc_id].append(position)
        
        return dict(positional_index)
    
    def _build_soundex_index(self) -> Dict[str, Dict[str, Set]]:
        """
        Build Soundex index: soundex_code -> {terms: set, docs: set}
        
        Returns:
            Dictionary mapping Soundex codes to terms and documents
        """
        soundex_index = defaultdict(lambda: {'terms': set(), 'docs': set()})
        
        for doc_id, document in enumerate(self.corpus):
            tokens = self.tokenize(document)
            
            for token in tokens:
                # Only process alphabetic tokens
                if token.isalpha():
                    code = self.soundex(token)
                    soundex_index[code]['terms'].add(token)
                    soundex_index[code]['docs'].add(doc_id)
        
        return dict(soundex_index)
    
    def _build_token_map(self) -> Dict[int, List[str]]:
        """
        Build map of original tokens for each document (for highlighting)
        
        Returns:
            Dictionary mapping doc_id to list of original tokens
        """
        token_map = {}
        for doc_id, document in enumerate(self.corpus):
            tokens = self.tokenize(document)
            token_map[doc_id] = tokens
        return token_map
    
    # ==================== ADVANCED: LEVENSHTEIN DISTANCE ====================
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """
        Calculate Levenshtein (Edit) Distance between two strings
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Edit distance (number of operations to transform s1 to s2)
        """
        if len(s1) < len(s2):
            return SearchEngine.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def find_closest_term(self, query_term: str, max_distance: int = 2) -> Optional[str]:
        """
        Find the closest matching term in the index using edit distance
        
        Args:
            query_term: The query term to match
            max_distance: Maximum edit distance to consider
            
        Returns:
            Closest matching term or None
        """
        best_match = None
        best_distance = max_distance + 1
        
        for term in self.inverted_index.keys():
            distance = self.levenshtein_distance(query_term, term)
            if distance < best_distance:
                best_distance = distance
                best_match = term
        
        return best_match if best_distance <= max_distance else None
    
    # ==================== ADVANCED: WILDCARD SEARCH ====================
    
    def wildcard_search(self, pattern: str) -> List[str]:
        """
        Find all terms matching wildcard pattern (e.g., comput*)
        
        Args:
            pattern: Pattern with * wildcard
            
        Returns:
            List of matching terms
        """
        # Convert wildcard pattern to regex
        regex_pattern = pattern.replace('*', '.*')
        regex_pattern = f"^{regex_pattern}$"
        
        matching_terms = []
        for term in self.inverted_index.keys():
            if re.match(regex_pattern, term):
                matching_terms.append(term)
        
        return matching_terms
    
    # ==================== RETRIEVAL ALGORITHMS ====================
    
    def boolean_retrieval(self, query: str) -> Tuple[List[int], List[str], Optional[str]]:
        """
        Boolean retrieval supporting AND, OR, NOT operators and wildcards
        
        Args:
            query: Boolean query string
            
        Returns:
            Tuple of (document IDs, processing steps, suggestion)
        """
        steps = []
        steps.append(f"Query: '{query}'")
        suggestion = None
        
        # Check for wildcards
        if '*' in query:
            steps.append("Detected wildcard search")
            
            # Extract wildcard term
            wildcard_terms = [term for term in query.split() if '*' in term]
            for wildcard_term in wildcard_terms:
                matching_terms = self.wildcard_search(wildcard_term)
                steps.append(f"Wildcard '{wildcard_term}' matches: {matching_terms}")
                
                # Replace wildcard with OR of matching terms
                if matching_terms:
                    replacement = ' OR '.join(matching_terms)
                    query = query.replace(wildcard_term, f"({replacement})")
                    steps.append(f"Expanded query: '{query}'")
        
        # Preprocess query
        processed = self.preprocess(query)
        stemmed_tokens = processed['stemmed_tokens']
        steps.append(f"Preprocessed terms: {stemmed_tokens}")
        
        result_docs = []
        
        # Handle AND operator
        if ' AND ' in query.upper():
            terms = [t.strip() for t in query.lower().split(' and ')]
            processed_terms = []
            
            for term in terms:
                term_processed = self.preprocess(term)
                if term_processed['stemmed_tokens']:
                    processed_terms.append(term_processed['stemmed_tokens'][0])
            
            steps.append(f"Boolean AND operation on: {processed_terms}")
            
            # Get documents for each term
            result_set = None
            for term in processed_terms:
                docs = set(self.inverted_index.get(term, []))
                steps.append(f"Term '{term}' found in documents: {sorted(docs)}")
                
                if result_set is None:
                    result_set = docs
                else:
                    result_set = result_set.intersection(docs)
            
            result_docs = sorted(list(result_set)) if result_set else []
            steps.append(f"Intersection (AND): {result_docs}")
        
        # Handle OR operator
        elif ' OR ' in query.upper():
            terms = [t.strip() for t in query.lower().split(' or ')]
            processed_terms = []
            
            for term in terms:
                term_processed = self.preprocess(term)
                if term_processed['stemmed_tokens']:
                    processed_terms.append(term_processed['stemmed_tokens'][0])
            
            steps.append(f"Boolean OR operation on: {processed_terms}")
            
            # Get documents for each term
            result_set = set()
            for term in processed_terms:
                docs = set(self.inverted_index.get(term, []))
                steps.append(f"Term '{term}' found in documents: {sorted(docs)}")
                result_set = result_set.union(docs)
            
            result_docs = sorted(list(result_set))
            steps.append(f"Union (OR): {result_docs}")
        
        # Handle NOT operator
        elif query.upper().startswith('NOT '):
            term = query[4:].strip().lower()
            term_processed = self.preprocess(term)
            
            if term_processed['stemmed_tokens']:
                processed_term = term_processed['stemmed_tokens'][0]
                steps.append(f"Boolean NOT operation on: '{processed_term}'")
                
                docs_with_term = set(self.inverted_index.get(processed_term, []))
                all_docs = set(range(len(self.corpus)))
                result_set = all_docs - docs_with_term
                
                result_docs = sorted(list(result_set))
                steps.append(f"Documents WITHOUT term '{processed_term}': {result_docs}")
        
        # Simple term search (OR of all terms)
        else:
            result_set = set()
            for term in stemmed_tokens:
                docs = self.inverted_index.get(term, [])
                steps.append(f"Term '{term}' found in documents: {docs}")
                result_set.update(docs)
            
            result_docs = sorted(list(result_set))
        
        # Spelling correction if no results
        if len(result_docs) == 0 and len(stemmed_tokens) > 0:
            steps.append("\n⚠️ No results found. Checking for spelling corrections...")
            for term in stemmed_tokens:
                closest = self.find_closest_term(term)
                if closest:
                    suggestion = closest
                    steps.append(f"Did you mean: '{closest}'?")
                    break
        
        steps.append(f"\nFinal Result: {len(result_docs)} document(s) retrieved")
        
        return result_docs, steps, suggestion
    
    def phrase_query(self, query: str) -> Tuple[List[int], List[str], Optional[str]]:
        """
        Phrase query: find documents with exact phrase (adjacent words)
        
        Args:
            query: Phrase query string
            
        Returns:
            Tuple of (document IDs, processing steps, suggestion)
        """
        steps = []
        steps.append(f"Phrase Query: '{query}'")
        suggestion = None
        
        # Preprocess query
        processed = self.preprocess(query)
        stemmed_tokens = processed['stemmed_tokens']
        steps.append(f"Preprocessed terms: {stemmed_tokens}")
        
        if len(stemmed_tokens) == 0:
            steps.append("No valid terms after preprocessing")
            return [], steps, None
        
        if len(stemmed_tokens) == 1:
            # Single term - just return documents containing it
            term = stemmed_tokens[0]
            docs = list(self.positional_index.get(term, {}).keys())
            steps.append(f"Single term '{term}' found in documents: {docs}")
            
            if len(docs) == 0:
                closest = self.find_closest_term(term)
                if closest:
                    suggestion = closest
                    steps.append(f"Did you mean: '{closest}'?")
            
            return sorted(docs), steps, suggestion
        
        # Multi-term phrase query
        first_term = stemmed_tokens[0]
        candidate_docs = self.positional_index.get(first_term, {})
        steps.append(f"First term '{first_term}' appears in documents: {list(candidate_docs.keys())}")
        
        result_docs = []
        
        # Check each candidate document
        for doc_id, positions in candidate_docs.items():
            steps.append(f"\nChecking Document {doc_id}:")
            
            # For each position of the first term
            for start_pos in positions:
                found = True
                steps.append(f"  - Checking position {start_pos} for '{first_term}'")
                
                # Check if subsequent terms appear at consecutive positions
                for i in range(1, len(stemmed_tokens)):
                    term = stemmed_tokens[i]
                    expected_pos = start_pos + i
                    
                    if term not in self.positional_index:
                        steps.append(f"    Term '{term}' not in index")
                        found = False
                        break
                    
                    if doc_id not in self.positional_index[term]:
                        steps.append(f"    Term '{term}' not in document {doc_id}")
                        found = False
                        break
                    
                    if expected_pos not in self.positional_index[term][doc_id]:
                        steps.append(f"    Term '{term}' not at position {expected_pos}")
                        found = False
                        break
                    else:
                        steps.append(f"    Term '{term}' found at position {expected_pos} ✓")
                
                if found:
                    result_docs.append(doc_id)
                    steps.append(f"  ✓ Phrase found in Document {doc_id} starting at position {start_pos}")
                    break  # Found in this document, move to next
        
        # Spelling correction if no results
        if len(result_docs) == 0:
            steps.append("\n⚠️ No results found. Checking for spelling corrections...")
            for term in stemmed_tokens:
                closest = self.find_closest_term(term)
                if closest:
                    suggestion = closest
                    steps.append(f"Did you mean: '{closest}'?")
                    break
        
        steps.append(f"\nFinal Result: {len(result_docs)} document(s) retrieved")
        
        return sorted(result_docs), steps, suggestion
    
    def soundex_search(self, query: str) -> Tuple[List[int], List[str], Dict[str, List[str]]]:
        """
        Soundex phonetic search
        
        Args:
            query: Search query
            
        Returns:
            Tuple of (document IDs, processing steps, matched terms)
        """
        steps = []
        steps.append(f"Query: '{query}'")
        
        # Tokenize query (no stemming for Soundex)
        tokens = self.tokenize(query)
        tokens = [t for t in tokens if t.isalpha()]  # Keep only alphabetic
        steps.append(f"Query tokens: {tokens}")
        
        result_docs = set()
        matched_terms = {}
        
        for token in tokens:
            code = self.soundex(token)
            steps.append(f"\nSoundex code for '{token}': {code}")
            
            if code in self.soundex_index:
                terms = sorted(list(self.soundex_index[code]['terms']))
                docs = sorted(list(self.soundex_index[code]['docs']))
                
                steps.append(f"Phonetically similar terms: {terms}")
                steps.append(f"Found in documents: {docs}")
                
                matched_terms[token] = terms
                result_docs.update(docs)
            else:
                steps.append(f"No phonetically similar terms found")
                matched_terms[token] = []
        
        steps.append(f"\nFinal Result: {len(result_docs)} document(s) retrieved")
        
        return sorted(list(result_docs)), steps, matched_terms
    
    # ==================== HIGHLIGHTING ====================
    
    def highlight_terms(self, doc_id: int, query_terms: List[str]) -> str:
        """
        Highlight query terms in document text
        
        Args:
            doc_id: Document ID
            query_terms: List of terms to highlight
            
        Returns:
            Document text with highlighted terms
        """
        text = self.corpus[doc_id]
        
        # Stem the query terms for matching
        stemmed_query_terms = set(self.stem_tokens(query_terms))
        
        # Get original tokens and their stemmed versions
        original_tokens = self.doc_tokens.get(doc_id, [])
        
        # Create a mapping of positions to highlight
        highlighted_text = text
        
        for query_term in query_terms:
            # Case-insensitive replacement with highlighting
            pattern = re.compile(re.escape(query_term), re.IGNORECASE)
            highlighted_text = pattern.sub(lambda m: f'<mark style="background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px;">{m.group()}</mark>', highlighted_text)
        
        return highlighted_text
    
    # ==================== UTILITY METHODS ====================
    
    def get_inverted_index(self) -> Dict[str, List[int]]:
        """Return the inverted index"""
        return self.inverted_index
    
    def get_positional_index(self) -> Dict[str, Dict[int, List[int]]]:
        """Return the positional index"""
        return self.positional_index
    
    def get_soundex_index(self) -> Dict[str, Dict[str, Set]]:
        """Return the Soundex index"""
        return self.soundex_index
    
    def get_corpus(self) -> List[str]:
        """Return the document corpus"""
        return self.corpus
    
    def get_total_terms(self) -> int:
        """Return total unique terms in inverted index"""
        return len(self.inverted_index)


# ==================== SAMPLE CORPUS ====================

SAMPLE_CORPUS = [
    "The quick brown fox jumps over the lazy dog",
    "Information retrieval is the process of obtaining information",
    "Python programming language is widely used for data science",
    "Machine learning algorithms can process large datasets",
    "Natural language processing involves computational linguistics",
    "Search engines use inverted indexes for fast retrieval",
    "Robert and Rania are working on the search engine project",
    "Herman Smith developed important algorithms for information systems",
    "Mohamed Muhammad and Mohammed are phonetically similar names",
    "The foxes are jumping quickly over the dogs in the garden"
]