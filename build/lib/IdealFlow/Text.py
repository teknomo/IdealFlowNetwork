# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 13:58:39 2021
TextProcessing.py v1
Last Udate: Dec 22, 2021

@author: Kardi Teknomo
"""
import os
import IdealFlow.Classifier as clf
import json
import re
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt



class NLP(clf.Classifier):
    """
    TextProcessing class for analyzing text data. Inherits from Classifier.

    This class provides methods for setting and retrieving text, loading text from a file, and performing
    text analysis, including sentence and word tokenization, word frequency counts, and text-to-matrix conversion.
    It us useful for basic Natural Language Understanding (NLU) class for basic NLP and NLU tasks.

    Attributes
    ----------
        :type text_id: str
        :param language: The language of the text.
        :type language: str
        :param model_name: The name of the model.
        :type model_name: str
    
    Methods
    -------
        set_text(text: str) -> None
            Sets the text for analysis.
        get_text() -> str
            Returns the text for analysis.
        load_text(file_name: str) -> None
            Loads text from a file into memory.
        prepareTextInput() -> tuple
            Converts text into a matrix X and vector y for classification.
        query(text: str) -> tuple
            Queries the IFN network with the given text and returns the predicted sentence and average probability.
        word_frequencies() -> dict
            Calculates word frequencies for the entire text.
        average_sentence_length() -> float
            Calculates the average number of words per sentence in the text.
        average_word_length() -> float
            Calculates the average word length in the text.
        unique_words_size() -> int
            Returns the number of unique words in the text.
    """
    def __init__(self, text_id: str ="", markov_order: int=1,  \
                 model_name: str = 'IFN', language: str = 'English', \
                intents_file: str = 'intents.json', \
                entity_patterns_file: str = 'entity_patterns.json'):  
        super().__init__(markov_order,text_id)      
        self.text_id = text_id
        self.language = language
        self.model_name = model_name
        self.text = ""
        self.vocabulary = set()
        self.accuracy = None
        self._training_data_size = 0       

        self.intents_file = os.path.join(os.path.dirname(__file__), intents_file)        
        self.entity_patterns_file = os.path.join(os.path.dirname(__file__), entity_patterns_file)
        self.intents = {}
        self.entity_patterns = {}
        self.load_patterns()
    

    @property
    def set_text(self, text: str) -> None:
        """
        Sets the text for analysis.

        Parameters
        ----------
        text : str
            The text to be analyzed.

        Example
        -------
        >>> import IdealFlow.Text as ift 
            >>> tp = ift.NLP("my_text")
        >>> tp.set_text = "This is a sample text."
        """
        self.text = text
    

    @property
    def get_text(self) -> str:
        """
        Returns the text for analysis.

        Returns
        -------
        str
            The current text set for analysis.

        Example
        -------
        >>> import IdealFlow.Text as ift 
        >>> tp = ift.NLP("my_text")
        >>> tp.get_text
        'This is a sample text.'
        """
        return self.text


    def load_text(self, file_name: str) -> None:
        """
        Loads text from a file into memory.

        Parameters
        ----------
        file_name : str
            The name of the file to load text from.

        Raises
        ------
        OSError
            If an error occurs during file reading.

        Example
        -------
        >>> import IdealFlow.Text as ift 
        >>> tp = ift.NLP("my_text")
        >>> tp.load_text('sample.txt')
        Loading Text from sample.txt
        Text was loaded successfully.
        """
        try:
            print(f"Loading Text from {file_name}")
            with open(file_name, 'r') as f:
                self.text=f.read()
                f.close()
            print("Text was loaded successfully.")    
        except OSError as e:
            print(f"An error occurred while loading text: {e}")
        
        
    """
    '
    '   PATTERNS RECOGNITION
    '
    """

    def load_patterns(self) -> None:
        """
        Load intent keywords and entity patterns from JSON files.

        **Example:**

        .. code-block:: python

            tp.load_patterns()
        """
        with open(self.intents_file, 'r') as f:
            self.intents = json.load(f)

        with open(self.entity_patterns_file, 'r') as f:
            self.entity_patterns = json.load(f)


    def recognize(self, text: str) -> dict:
        """
        Generate predictions for the input text, such as intent classification or entity recognition.

        :param text: The input text.
        :type text: str
        :return: Predictions including intent and entities.
        :rtype: dict

        **Example:**

        .. code-block:: python
            >>> import IdealFlow.Text as ift 
            >>> tp = ift.NLP("my_text")
            >>> recognitions = tp.recognize("Book a flight to New York")
        """
        intent = self.intent_recognition(text)
        entities = self.entity_recognition(text)
        return {'intent': intent, 'entities': entities}
        # # Simple intent classification based on keywords
        # tokens = self.tokenize(text)
        # intents = {
        #     'book_flight': ['book', 'flight', 'ticket'],
        #     'weather_query': ['weather', 'temperature', 'forecast'],
        #     'greeting': ['hello', 'hi', 'hey']
        # }

        # predicted_intent = 'unknown'
        # for intent, keywords in intents.items():
        #     if any(token in keywords for token in tokens):
        #         predicted_intent = intent
        #         break

        # # Simple entity recognition based on capitalization
        # entities = []
        # for token in tokens:
        #     if token.istitle():
        #         entities.append({'entity': token, 'type': 'ProperNoun'})

        # return {'intent': predicted_intent, 'entities': entities}

    
    def evaluate(self, test_data: list) -> dict:
        """
        Assess the performance of the model using test data, returning metrics like accuracy.

        :param test_data: List of tuples (text, expected_intent).
        :type test_data: list
        :return: Evaluation metrics.
        :rtype: dict

        **Example:**

        .. code-block:: python
            >>> import IdealFlow.Text as ift 
            >>> tp = ift.NLP("my_text")
            >>> test_data = [("Book a flight", "book_flight"), ...]
            >>> metrics = tp.evaluate(test_data)
        """
        correct_intent = 0
        total = len(test_data)
        for item in test_data:
            text = item['text']
            expected_intent = item['intent']
            prediction = self.intent_recognition(text)
            if prediction == expected_intent:
                correct_intent += 1
        accuracy = correct_intent / total if total > 0 else 0
       
        self.accuracy = accuracy
        return {'accuracy': accuracy}



    def update_model(self, new_data: list) -> None:
        """
        Train the NLU model with new data and allow users to associate patterns.

        :param new_data: List of new training data dictionaries with 'text' and 'intent'.
        :type new_data: list

        **Example:**

        .. code-block:: python

            new_data = [{"text": "Schedule a meeting", "intent": "schedule_meeting"}]
            tp.update_model(new_data)
        """
        for item in new_data:
            text = item['text']
            intent = item['intent']
            # tokens = self.tokenize(text.lower())
            tokens = self.lemmatize(self.tokenize(text.lower()))
            self.vocabulary.update(tokens)
            self._training_data_size += len(tokens)
            if intent not in self.intents:
                self.intents[intent] = []
            self.intents[intent].extend(tokens)
            # Remove duplicates
            self.intents[intent] = list(set(self.intents[intent]))
        
        self.save_intent()

    # def update_model(self, new_data: list) -> None:
    #     """
    #     Fine-tune the NLU model with new training data to improve accuracy or adapt to new domains.

    #     :param new_data: List of new training data tuples (text, intent).
    #     :type new_data: list

    #     **Example:**

    #     .. code-block:: python

    #         new_data = [("I want to book a flight", "book_flight"), ...]
    #         tp.update_model(new_data)
    #     """
    #     # For simplicity, we'll just update the vocabulary
    #     for text, intent in new_data:
    #         tokens = self.tokenize(text)
    #         self.vocabulary.update(tokens)
    #         self.training_data_size += len(tokens)


    def save_model(self, filepath: str) -> None:
        """
        Save the current state of the NLU model to a file for later use or distribution.

        :param filepath: The file path to save the model.
        :type filepath: str

        **Example:**

        .. code-block:: python

            tp.save_model('model.npz')
        """
        np.savez(filepath, vocabulary=list(self.vocabulary))


    def load_model(self, filepath: str) -> None:
        """
        Load a pre-trained model from a file.

        :param filepath: The file path to load the model from.
        :type filepath: str

        **Example:**

        .. code-block:: python

            tp.load_model('model.npz')
        """
        data = np.load(filepath, allow_pickle=True)
        self.vocabulary = set(data['vocabulary'])



    """
    '
    '   INTENTS RECOGNITION
    '
    """

    def add_intent(self, intent_name: str, keywords: list) -> None:
        """
        Add a new intent with associated keywords.

        :param intent_name: The name of the intent.
        :type intent_name: str
        :param keywords: List of keywords associated with the intent.
        :type keywords: list

        **Example:**

        .. code-block:: python

            tp.add_intent('schedule_meeting', ['schedule', 'meeting', 'appointment'])
        """
        self.intents[intent_name] = keywords        
        self.save_intent()
    

    def save_intent(self) -> None:
        """
        Save current intent keywords patterns to JSON files.

        **Example:**

        .. code-block:: python

            tp. save_intents()
        """
        try:
            # print(f"Saving intents to {self.intents_file}")
            with open(self.intents_file, 'w') as f:
                json.dump(self.intents, f, indent=4)
            print("Intents saved successfully.")    
        except OSError as e:
            print(f"An error occurred while saving intents: {e}")

    
    def intent_recognition(self, text: str) -> str:
        """
        Identify the user's intent in the input text.

        :param text: The input text.
        :type text: str
        :return: Predicted intent.
        :rtype: str

        **Example:**

        .. code-block:: python

            intent = tp.intent_recognition("Book a flight to New York")
        """
        tokens = self.lemmatize(self.tokenize(text.lower()))
        # tokens = self.tokenize(text.lower())
        intent_scores = {intent: 0 for intent in self.intents}
        for intent, keywords in self.intents.items():
            lemmatized_keywords = [self.lemmatize([kw])[0] for kw in keywords]
            for token in tokens:
                if token in lemmatized_keywords:
                    intent_scores[intent] += 1

        # Select the intent with the highest score
        predicted_intent = max(intent_scores, key=intent_scores.get)
        if intent_scores[predicted_intent] == 0:
            predicted_intent = 'unknown'

        return predicted_intent



    """
    '
    '   ENTITY RECOGNITION
    '
    """

    def add_entity_pattern(self, entity_type: str, pattern: str, description: str = "") -> None:
        """
        Add a new entity pattern.

        :param entity_type: The type of the entity.
        :type entity_type: str
        :param pattern: The regex pattern for the entity.
        :type pattern: str

        **Example:**

        .. code-block:: python

            tp.add_entity_pattern('Email', r'\\b[\\w.-]+@[\\w.-]+\\.\\w{2,4}\\b')
        """
        self.entity_patterns[entity_type] = {
            'pattern': pattern,
            'description': description  # Include description in the dictionary
        }
        self.save_entity()


    def save_entity(self) -> None:
        """
        Save current entity patterns to JSON files.

        **Example:**

        .. code-block:: python

            tp. save_entity()pip list
        """
        try:
            # print(f"Saving entity patterns to {self.entity_patterns_file}")
            with open(self.entity_patterns_file, 'w') as f:
                json.dump(self.entity_patterns, f, indent=4)
            print("Entity patterns saved successfully.")                
        except OSError as e:
            print(f"An error occurred while saving entity patterns: {e}")


    def entity_recognition(self, text: str) -> list:
        """
        Identify entities in the input text.

        :param text: The input text.
        :type text: str
        :return: List of identified entities with their types.
        :rtype: list

        **Example:**

        .. code-block:: python

            entities = tp.entity_recognition("Book a flight to New York on September 21st")
        """
        entities = []
        for entity_type, entity_info in self.entity_patterns.items():
            pattern = entity_info['pattern']  # Access the 'pattern' key
            matches = re.findall(pattern, text)
            for match in matches:
                entities.append({'entity': match, 'type': entity_type})

        # Remove duplicates
        entities = [dict(t) for t in {tuple(d.items()) for d in entities}]
        return entities



    """
    '
    '   TOKENIZATION FROM TEXT TO TOKEN
    '
    """

    def get_paragraphs(self) -> list:
        """
        Separate the text into a list of paragraphs.

        :return: List of paragraphs.
        :rtype: list

        **Example:**

        .. code-block:: python

            >>> import IdealFlow.Text as ift 
            >>> tp = ift.NLP("my_text")
            >>> tp.text = "Paragraph one.\n\nParagraph two."
            >>> paragraphs = tp.get_paragraphs()
        """
        paragraphs = self.text.strip().split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]


    @staticmethod
    def text_to_paragraphs(text):
        '''
        return list of paragraphs from the given text string
        '''
        paragraphs=[]
        para=text.split('\n\n')
        for p in para:
            p = p.strip()
            if len(p) > 0:
                paragraphs.append(p)
        return paragraphs
    

    def get_sentences(self, paragraph: str) -> list:
        """
        Separate a paragraph into a list of sentences.

        :param paragraph: The paragraph to split.
        :type paragraph: str
        :return: List of sentences.
        :rtype: list

        **Example:**

        .. code-block:: python

            sentences = tp.get_sentences(paragraph)
        """
        # Simple sentence tokenizer using regex
        sentence_endings = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_endings.split(paragraph)
        return [s.strip() for s in sentences if s.strip()]

    
    def tokenize(self, text: str) -> list:
        """
        Tokenize a text into a list of tokens (words). 
        Separate a sentence into a list of tokens (words).
        return the list of tokens (= words, punctuation) from a sentence

        :param text: The text to tokenize.
        :type text: str
        :return: List of tokens.
        :rtype: list

        **Example:**

        .. code-block:: python

            tokens = tp.tokenize("Hello, world!")
        """             
        tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)
        return tokens
    

    def text_to_sentences(self):
        """
        return list of sentences from the given text string
        """
        list_sentences =[]  
        paragraphs = self.get_paragraphs()
        for paragraph in paragraphs:
            sentences = self.get_sentences(paragraph)
            for sentence in sentences:
                list_sentences.append(sentence)
        
        return list_sentences


    def sentence_to_tokens(self, sentence):
        '''
        return the list of tokens (= words, punctuation)
        from a sentence
        '''
        list_tokens=[]
        tokens = self.tokenize(sentence)
        for token in tokens:
            list_tokens.append(token)
        return list_tokens


    def sentences_to_text(self, sentences: list) -> str:
        """
        Convert a list of sentences back into text.

        :param sentences: The list of sentences.
        :type sentences: list
        :return: The combined text.
        :rtype: str

        **Example:**

        .. code-block:: python

            text = tp.sentences_to_text(sentences)
        """
        return ' '.join(sentences)


    def remove_stopwords(self, tokens: list) -> list:
        """
        Remove stop words from a list of tokens.

        :param tokens: The list of tokens.
        :type tokens: list
        :return: Tokens without stop words.
        :rtype: list

        **Example:**

        .. code-block:: python

            tokens = tp.tokenize(text)
            filtered_tokens = tp.remove_stopwords(tokens)
        """
        # A simple list of common English stop words
        stop_words = set([
            'a', 'an', 'the', 'and', 'or', 'if', 'to', 'of', 'in', 'on',
            'for', 'with', 'as', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'at', 'by', 'from', 'up', 'down', 'not', 'that', 'this',
            'but', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'into'
        ])
        return [token for token in tokens if token.lower() not in stop_words]

    
    """
    '
    '   DETOKENIZATION FROM TOKEN TO TEXT
    '
    """

    def detokenize(self, tokens: list) -> str:
        """
        Detokenize a list of tokens back into a string.
        Returns a string with proper spacing and punctuation, following standard detokenization rules.
        Handles code blocks, LaTeX formulas, markdown syntax, contractions, and punctuation.

        :param tokens: The list of tokens.
        :type tokens: list
        :return: The detokenized string.
        :rtype: str

        **Example:**

        .. code-block:: python

            text = tp.detokenize(tokens)
        """
        try:
            text = ''
            prev_token = None
            for token in tokens:
                # Handle special tokens (code blocks, inline code, LaTeX formulas)
                if token.startswith('```') and token.endswith('```'):
                    # Code block
                    if text and text[-1] != '\n':
                        text += '\n'
                    text += token
                    if not text.endswith('\n'):
                        text += '\n'
                elif token.startswith('`') and token.endswith('`'):
                    # Inline code
                    if text and text[-1] != ' ':
                        text += ' '
                    text += token
                elif (token.startswith('$$') and token.endswith('$$')) or (token.startswith('$') and token.endswith('$')):
                    # LaTeX formulas
                    if text and text[-1] != ' ':
                        text += ' '
                    text += token
                elif token.startswith('#'):
                    # Markdown headings
                    if text and text[-1] != '\n':
                        text += '\n'
                    text += token
                elif token in {'.', ',', '!', '?', ':', ';', '%', ')', ']', '}', '...', "'", '"'}:
                    # Don't add space before these tokens
                    text += token
                elif token in {'(', '[', '{', '$'}:
                    # Don't add space after these tokens
                    if text and text[-1] != ' ' and text[-1] not in {'\n', ' '}:
                        text += ' '
                    text += token
                elif token == "'s" or token == "n't" or (token.startswith("'") and len(token) > 1):
                    # Attach contractions directly
                    text += token
                elif token == '-':
                    # Handle hyphens (e.g., 'state-of-the-art')
                    text += token
                else:
                    # Add space before the token
                    if text and text[-1] not in {' ', '(', '[', '{', '-', '/', '\n'}:
                        text += ' '
                    text += token
                prev_token = token

            # Cleanup text by removing extra quotes and backslashes
            text = text.replace("``", '"').replace("''", '"')
            text = text.replace('\"', '"')
            text = text.replace("\\", "")
            return text.strip()
        except TypeError as e:
            print(f"An error occurred while detokenizing: {e}. Perhaps the tokens are None.")
            return ""


    """
    '
    '   LEMMATIZATION: THE ROOT WORD
    '
    """

    def lemmatize(self, tokens: list) -> list:
        """
        Lemmatize a list of tokens to their base forms.

        :param tokens: The list of tokens.
        :type tokens: list
        :return: List of lemmatized tokens.
        :rtype: list

        **Example:**

        .. code-block:: python

            lemmatized_tokens = tp.lemmatize(['running', 'jumps', 'easily'])
        """
        # Simple rule-based lemmatizer
        lemmas = []
        for token in tokens:
            lemma = token.lower()
            if lemma.endswith('ing') or lemma.endswith('ed'):
                lemma = lemma.rstrip('ing').rstrip('ed')
            elif lemma.endswith('s') and not lemma.endswith('ss'):
                lemma = lemma.rstrip('s')
            lemmas.append(lemma)
        return lemmas    
    

    """
    '
    '   PART-OF-SPEECH
    '
    """


    def parse_sentence(self, sentence: str) -> dict:
        """
        Parse the sentence to identify grammatical components  (such as  subject, predicate, and object) using a simple dependency parsing approach.
        

        :param sentence: The sentence to parse.
        :type sentence: str
        :return: A dictionary representing the parse tree.
        :rtype: dict

        **Example:**

        .. code-block:: python

            parsed = tp.parse_sentence("The cat eats fish.")
            parse_tree = tp.parse_sentence("The quick brown fox jumps over the lazy dog.")
        """
        # Tokenize and tag parts of speech using simple regex-based rules
        tokens = self.tokenize(sentence)
        pos_tags = self.simple_pos_tagging(tokens)
        parse_tree = self.build_dependency_tree(tokens, pos_tags)
        return parse_tree
        # # Very basic and naive parsing using regex
        # pattern = r'^(?P<subject>\w+)\s+(?P<predicate>\w+)\s+(?P<object>\w+)\.?$'
        # match = re.match(pattern, sentence)
        # if match:
        #     return match.groupdict()
        # else:
        #     return {'subject': None, 'predicate': None, 'object': None}


    def simple_pos_tagging(self, tokens: list) -> list:
        """
        Perform simple part-of-speech tagging.

        :param tokens: The list of tokens.
        :type tokens: list
        :return: List of tuples with token and POS tag.
        :rtype: list

        **Example:**

        .. code-block:: python

            pos_tags = tp.simple_pos_tagging(['The', 'cat', 'sat'])
        """
        # Very basic and naive POS tagging using suffixes
        pos_tags = []
        for token in tokens:
            lower_token = token.lower()
            if re.match(r'.*ing$', lower_token):
                pos_tags.append((token, 'VBG'))  # Verb Gerund
            elif re.match(r'.*ed$', lower_token):
                pos_tags.append((token, 'VBD'))  # Verb Past Tense
            elif re.match(r'.*ly$', lower_token):
                pos_tags.append((token, 'RB'))   # Adverb
            elif re.match(r'.*ness$', lower_token):
                pos_tags.append((token, 'NN'))   # Noun
            elif re.match(r'.*ment$', lower_token):
                pos_tags.append((token, 'NN'))   # Noun
            elif lower_token in ['the', 'a', 'an']:
                pos_tags.append((token, 'DT'))   # Determiner
            elif lower_token in ['he', 'she', 'it', 'they', 'we', 'i', 'you']:
                pos_tags.append((token, 'PRP'))  # Pronoun
            elif lower_token in ['is', 'are', 'was', 'were', 'be', 'been', 'being']:
                pos_tags.append((token, 'VB'))   # Verb
            elif re.match(r'.*s$', lower_token) and lower_token != 'is':
                pos_tags.append((token, 'NNS'))  # Noun Plural
            else:
                pos_tags.append((token, 'NN'))   # Noun as default
        return pos_tags


    def build_dependency_tree(self, tokens: list, pos_tags: list) -> dict:
        """
        Build a simple dependency tree based on POS tags.

        :param tokens: The list of tokens.
        :type tokens: list
        :param pos_tags: The list of POS tags.
        :type pos_tags: list
        :return: A dependency tree represented as a dictionary.
        :rtype: dict

        **Example:**

        .. code-block:: python

            tree = tp.build_dependency_tree(tokens, pos_tags)
        """
        tree = defaultdict(list)
        for i, (token, pos) in enumerate(pos_tags):
            if pos.startswith('V'):  # Verb
                tree['predicate'].append(token)
            elif pos.startswith('N') or pos == 'PRP':  # Noun or Pronoun
                if 'predicate' in tree:
                    tree['object'].append(token)
                else:
                    tree['subject'].append(token)
            elif pos == 'DT':  # Determiner
                continue  # Skip determiners
            else:
                tree['modifiers'].append(token)
        return dict(tree)



    """
    '
    '   NATURAL LANGUAGE ANALYSIS
    '
    """

    def bag_of_words(self, tokens: list) -> dict:
        """
        Create a bag-of-words representation (word frequency dictionary) from a list of tokens.

        :param tokens: The list of tokens.
        :type tokens: list
        :return: Word frequency dictionary.
        :rtype: dict

        **Example:**

        .. code-block:: python

            bow = tp.bag_of_words(tokens)
        """
        return dict(Counter(tokens))


    def ngrams(self, tokens: list, n: int = 2) -> list:
        """
        Generate n-grams (e.g., bigrams, trigrams) from a list of tokens.

        :param tokens: The list of tokens.
        :type tokens: list
        :param n: The number of items in each n-gram.
        :type n: int
        :return: List of n-grams.
        :rtype: list

        **Example:**

        .. code-block:: python

            bigrams = tp.ngrams(tokens, n=2)
        """
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    

    def plot_word_frequencies(self, top_n: int = 10) -> None:
        """
        Plot the top N word frequencies in the text.

        :param top_n: Number of top words to plot.
        :type top_n: int

        **Example:**

        .. code-block:: python

            tp.plot_word_frequencies(top_n=15)
        """
        word_counts = Counter(self.tokenize(self.text.lower()))
        most_common = word_counts.most_common(top_n)
        words, counts = zip(*most_common)
        plt.bar(words, counts)
        plt.xticks(rotation=45)
        plt.title('Top {} Word Frequencies'.format(top_n))
        plt.show()


    def vectorize(self, tokens: list) -> np.array:
        """
        Transform tokens into numerical vectors suitable for processing by machine learning models.

        :param tokens: The list of tokens.
        :type tokens: list
        :return: Numerical vector representation of tokens.
        :rtype: np.array

        **Example:**

        .. code-block:: python

            vector = tp.vectorize(tokens)
        """
        # Simple one-hot encoding
        self.vocabulary.update(tokens)
        vocab_list = list(self.vocabulary)
        vector = np.zeros(len(vocab_list))
        token_counts = Counter(tokens)
        for token, count in token_counts.items():
            index = vocab_list.index(token)
            vector[index] = count
        return vector


    def summarize(self, text: str, n_sentences: int = 1) -> str:
        """
        Summarize the input text.

        :param text: The text to summarize.
        :type text: str
        :param n_sentences: Number of sentences to include in the summary.
        :type n_sentences: int
        :return: The summarized text.
        :rtype: str

        **Example:**

        .. code-block:: python

            summary = tp.summarize(long_text, n_sentences=2)
        """
        # Simple summarization by selecting the first n sentences
        sentences = self.get_sentences(text)
        summary = ' '.join(sentences[:n_sentences])
        return summary
    
    
    def normalize_text(self) -> str:
        """
        Normalize the text by lowercasing and removing punctuation.

        :return: Normalized text.
        :rtype: str

        **Example:**

        .. code-block:: python

            normalized_text = tp.normalize_text()
        """
        return re.sub(r'[^\w\s]', '', self.text.lower())
   

    def word_frequencies(self) -> dict:
        """
        Calculates word frequencies for the entire text.

        Returns
        -------
        dict
            A dictionary of word frequencies.

        Example
        -------
        >>> import IdealFlow.Text as ift 
        >>> tp = ift.NLP("my_text")
        >>> tp.word_frequencies()
        {'word1': 5, 'word2': 3}
        """
        normalized_text = self.normalize_text()
        tokens = normalized_text.split()
        return Counter(tokens)


    def average_sentence_length(self) -> float:
        """
        Calculates the average number of words per sentence in the text.

        Returns
        -------
        float
            The average number of words per sentence.

        Example
        -------
        >>> import IdealFlow.Text as ift 
        >>> tp = ift.NLP("my_text")
        >>> tp.average_sentence_length()
        10.5
        """
        paragraphs = self.get_paragraphs()
        sentences = [self.get_sentences(p) for p in paragraphs]
        all_sentences = [s for sublist in sentences for s in sublist]
        total_words = sum(len(self.tokenize(s)) for s in all_sentences)
        return total_words / len(all_sentences) if all_sentences else 0


    def average_word_length(self) -> float:
        """
        Calculates the average length of words in the text.

        Returns
        -------
        float
            The average word length.

        Example
        -------
        >>> import IdealFlow.Text as ift 
        >>> tp = ift.NLP("my_text")
        >>> tp.average_word_length()
        4.75
        """
        normalized_text = self.normalize_text()
        tokens = normalized_text.split()
        total_chars = sum(len(token) for token in tokens)
        return total_chars / len(tokens) if tokens else 0
    

    def unique_words_size(self) -> int:
        """
        Calculates the number of unique words in the text.

        Returns
        -------
        int
            The number of unique words in the text.

        Example
        -------
        >>> import IdealFlow.Text as ift 
        >>> tp = ift.NLP("my_text")
        >>> tp.unique_words_size()
        120
        """
        normalized_text = self.normalize_text()
        tokens = set(normalized_text.split())
        return len(tokens)        


    @property
    def vocabulary_size(self) -> int:
        """
        The size of the vocabulary the model can recognize.

        :return: Vocabulary size.
        :rtype: int

        **Example:**

        .. code-block:: python

            size = tp.vocabulary_size
        """
        return len(self.vocabulary)


    @property
    def training_data_size(self) -> int:
        """
        The amount of data used to train the model (number of tokens).

        :return: Training data size.
        :rtype: int

        **Example:**

        .. code-block:: python

            size = tp.training_data_size
        """
        return self._training_data_size

    @training_data_size.setter
    def training_data_size(self, value: int) -> None:
        """
        Sets the size of the training data.

        Parameters
        ----------
        value : int
            The size of the training data.

        Example
        -------
        >>> import IdealFlow.Text as ift 
        >>> tp = ift.NLP("my_text")
        >>> tp.training_data_size = 100
        """
        self._training_data_size = value

    
    def prepareTextInput(self) -> tuple:
        """
        Converts text into a matrix X and vector y for classification.

        Each sentence is treated as a trajectory, and each word as a node. The vector y contains repeated category labels. category = text_id to be set when we initiate the class.

        Returns
        -------
        tuple
            A tuple (X, y) where X is the matrix of tokenized sentences and y is the vector of category labels.

        Example
        -------
        >>> import IdealFlow.Text as ift 
            >>> tp = ift.NLP("my_text")
        >>> X, y = tp.prepareTextInput()
        >>> X  # Tokenized sentences
        [['word1', 'word2'], ['word3', 'word4']]
        >>> y  # Category labels
        ['category', 'category']
        """
        sentences=self.text_to_sentences()
        category = self.text_id
        mR=len(sentences)
        self.variables=None
        y=[category]*mR
        X = [self.sentence_to_tokens(sentence) for sentence in sentences]
       
        return X,y
    

    def predict_text_category(self,text: str) -> tuple:
        """
        Predicts the category for the provided text.

        Parameters
        ----------
            text : str text input that you want which text belongs to       

        Returns
        -------
            tuple: The name of the IFN with the maximum entropy and the percentage of confidence (float).       

        Example:
            >>> import IdealFlow.Text as ift 
            >>> tp = ift.NLP("my_text")
            >>> X="In particular, NLP is used to program"
            >>> print("predict result",tp.find_category(X))
        
        """
        trajectory = self.sentence_to_tokens(text)
        code=self.trajVarVal2Code(trajectory)
        return super().predict(code)


    def query(self, text: str) -> tuple:
        """
        Queries the IFN network with the given text and returns the predicted sentence and average probability.

        Parameters
        ----------
        text : str
            The input text to query.

        Returns
        -------
        tuple
            A tuple (sentence, avg_prob) where sentence is the predicted sentence and avg_prob is the average probability.

        Example
        -------
            >>> import IdealFlow.Text as ift 
            >>> tp = ift.NLP("my_text")
            >>> tp.query("This is a query.")
            ('Predicted sentence', 0.85)
        """
        ifn=self.__searchIFNs__(self.text_id)
        if str(ifn.adjList)!='{}':
            lst=self.tokenize(text)
            full_path, avg_prob = ifn.query(self.trajVarVal2Code(lst), method='min')
            lstTokens = self.trajCode2VarVal(full_path)
            sentence = self.detokenize(lstTokens)
            return sentence, avg_prob

# END OF class Table_Classifier


if __name__=='__main__':
    import IdealFlow.Text as ift     # import package.module as alias
    tp=ift.NLP("my_text")
    
    
    
    

    
    