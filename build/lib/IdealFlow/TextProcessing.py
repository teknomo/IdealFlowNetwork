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
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import time
import asyncio

class TextProcessing():
    """
    Natural Language Understanding (NLU) class for basic NLP and NLU tasks.

    :param text_name: The text to be processed.
    :type text_name: str
    :param language: The language of the text.
    :type language: str
    :param model_name: The name of the model.
    :type model_name: str
    """
    def __init__(self, markov_order: int=1, text_id: str ="", \
                 model_name: str = 'IFN', language: str = 'English', \
                intents_file: str = 'intents.json', \
                entity_patterns_file: str = 'entity_patterns.json'):        
        self.text_id = text_id
        self.language = language
        self.model_name = model_name
        self.text = ""
        self.vocabulary = set()
        self.accuracy = None
        self._training_data_size = 0
        self.markovOrder=markov_order
        self.ifnc=clf.Classifier(markov_order,name=text_id)

        self.intents_file = os.path.join(os.path.dirname(__file__), intents_file)        
        self.entity_patterns_file = os.path.join(os.path.dirname(__file__), entity_patterns_file)
        self.intents = {}
        self.entity_patterns = {}
        self.load_patterns()
    

    @property
    def set_text(self,text):
        '''
        set the text for analysis
        '''
        self.text = text
    

    @property
    def get_text(self):
        '''
        get the text for analysis
        '''
        return self.text


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


    def save_intent(self) -> None:
        """
        Save current intent keywords patterns to JSON files.

        **Example:**

        .. code-block:: python

            tp. save_intents()
        """
        try:
            print(f"Saving intents to {self.intents_file}")
            with open(self.intents_file, 'w') as f:
                json.dump(self.intents, f, indent=4)
            print("Intents saved successfully.")    
        except OSError as e:
            print(f"An error occurred while saving intents: {e}")


    def save_entity(self) -> None:
        """
        Save current entity patterns to JSON files.

        **Example:**

        .. code-block:: python

            tp. save_entity()
        """
        try:
            print(f"Saving entity patterns to {self.entity_patterns_file}")
            with open(self.entity_patterns_file, 'w') as f:
                json.dump(self.entity_patterns, f, indent=4)
            print("Entity patterns saved successfully.")                
        except OSError as e:
            print(f"An error occurred while saving entity patterns: {e}")


    # async def save_file_async(self, file_path, data):
    #     await asyncio.sleep(0.1)  # Simulate async delay
    #     with open(file_path, 'w') as f:
    #         json.dump(data, f, indent=4)
    #         f.flush()
    #         os.fsync(f.fileno())


    # async def save_patterns(self) -> None:
    #     """
    #     Save current intent keywords and entity patterns to JSON files.

    #     **Example:**

    #     .. code-block:: python

    #         asyncio.run(tp.save_patterns())
    #     """
    #     try:
    #         # with open(self.intents_file, 'w') as f:
    #         #     json.dump(self.intents, f, indent=4)
    #         #     f.flush()
    #         #     os.fsync(f.fileno())
    #         # time.sleep(0.1)  # Introduce a short delay
    #         await self.save_file_async(self.intents_file, self.intents)
    #         print("Intents saved successfully.")
            
    #     except OSError as e:
    #         print(f"An error occurred while saving intents: {e}")
    #     # time.sleep(0.1)  # Introduce a short delay between operations
    #     try:
    #         # with open(self.entity_patterns_file, 'w') as f:
    #         #     json.dump(self.entity_patterns, f, indent=4)
    #         #     f.flush()
    #         #     os.fsync(f.fileno())
    #         # time.sleep(0.1)  # Introduce a short delay   
    #         await self.save_file_async(self.entity_patterns_file, self.entity_patterns) 
    #         print("Entity patterns saved successfully.")
            
    #     except OSError as e:
    #         print(f"An error occurred while saving entity patterns: {e}")


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
        # asyncio.run(self.save_patterns())
        self.save_entity()


    def add_entity_pattern(self, entity_type: str, pattern: str) -> None:
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
        self.entity_patterns[entity_type] = pattern
        # asyncio.run(self.save_patterns())
        self.save_entity()


    def get_paragraphs(self) -> list:
        """
        Separate the text into a list of paragraphs.

        :return: List of paragraphs.
        :rtype: list

        **Example:**

        .. code-block:: python

            nlu = NLU(text_name='sample.txt')
            tp.text = "Paragraph one.\n\nParagraph two."
            paragraphs = tp.get_paragraphs()
        """
        paragraphs = self.text.strip().split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]

    
    def text_to_paragraphs(self,text):
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


    def parse_sentence(self, sentence: str) -> dict:
        """
        Parse the sentence to identify subject, predicate, and object.

        :param sentence: The sentence to parse.
        :type sentence: str
        :return: A dictionary with 'subject', 'predicate', 'object'.
        :rtype: dict

        **Example:**

        .. code-block:: python

            parsed = tp.parse_sentence("The cat eats fish.")
        """
        # Very basic and naive parsing using regex
        pattern = r'^(?P<subject>\w+)\s+(?P<predicate>\w+)\s+(?P<object>\w+)\.?$'
        match = re.match(pattern, sentence)
        if match:
            return match.groupdict()
        else:
            return {'subject': None, 'predicate': None, 'object': None}
        
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
        # Remove punctuation and split by whitespace
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    

    def detokenize(self, tokens: list) -> str:
        """
        Detokenize a list of tokens back into a string.
        return a sentence from list of tokens

        :param tokens: The list of tokens.
        :type tokens: list
        :return: The detokenized string.
        :rtype: str

        **Example:**

        .. code-block:: python

            text = tp.detokenize(['hello', 'world'])
        """
        sentence = ' '.join(tokens)
        sentence=sentence.replace('\"', '"')
        sentence=sentence.replace("''", "")
        sentence=sentence.replace('\"', "")
        sentence=sentence.replace('``', "")
        sentence=sentence.replace(" \ ", "")
        return sentence
        
    
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
        tokens = self.tokenize(text.lower())
        intent_scores = {intent: 0 for intent in self.intents}
        for intent, keywords in self.intents.items():
            for token in tokens:
                if token in keywords:
                    intent_scores[intent] += 1

        # Select the intent with the highest score
        predicted_intent = max(intent_scores, key=intent_scores.get)
        if intent_scores[predicted_intent] == 0:
            predicted_intent = 'unknown'

        return predicted_intent

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
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                entities.append({'entity': match, 'type': entity_type})

        # Remove duplicates
        entities = [dict(t) for t in {tuple(d.items()) for d in entities}]
        return entities
    
    def predict(self, text: str) -> dict:
        """
        Generate predictions for the input text, such as intent classification or entity recognition.

        :param text: The input text.
        :type text: str
        :return: Predictions including intent and entities.
        :rtype: dict

        **Example:**

        .. code-block:: python

            predictions = tp.predict("Book a flight to New York")
        """
        # Simple intent classification based on keywords
        tokens = self.tokenize(text)
        intents = {
            'book_flight': ['book', 'flight', 'ticket'],
            'weather_query': ['weather', 'temperature', 'forecast'],
            'greeting': ['hello', 'hi', 'hey']
        }

        predicted_intent = 'unknown'
        for intent, keywords in intents.items():
            if any(token in keywords for token in tokens):
                predicted_intent = intent
                break

        # Simple entity recognition based on capitalization
        entities = []
        for token in tokens:
            if token.istitle():
                entities.append({'entity': token, 'type': 'ProperNoun'})

        return {'intent': predicted_intent, 'entities': entities}

    def evaluate(self, test_data: list) -> dict:
        """
        Assess the performance of the model using test data, returning metrics like accuracy.

        :param test_data: List of tuples (text, expected_intent).
        :type test_data: list
        :return: Evaluation metrics.
        :rtype: dict

        **Example:**

        .. code-block:: python

            test_data = [("Book a flight", "book_flight"), ...]
            metrics = tp.evaluate(test_data)
        """
        correct = 0
        total = len(test_data)
        for text, expected_intent in test_data:
            prediction = self.predict(text)
            if prediction['intent'] == expected_intent:
                correct += 1
        accuracy = correct / total if total > 0 else 0
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
            tokens = self.tokenize(text.lower())
            self.vocabulary.update(tokens)
            self._training_data_size += len(tokens)
            if intent not in self.intents:
                self.intents[intent] = []
            self.intents[intent].extend(tokens)
            # Remove duplicates
            self.intents[intent] = list(set(self.intents[intent]))
        # asyncio.run(self.save_patterns())
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
    
    # Additional basic NLP methods

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
   

    def word_frequencies(self):
        """
        Calculate word frequencies for the entire text.
        """
        normalized_text = self.normalize_text()
        tokens = normalized_text.split()
        return Counter(tokens)

    def vocabulary_size(self):
        """
        Calculate the number of unique words in the text.
        """
        normalized_text = self.normalize_text()
        tokens = set(normalized_text.split())
        return len(tokens)

    def average_sentence_length(self):
        """
        Calculate the average number of words per sentence in the text.
        """
        paragraphs = self.get_paragraphs()
        sentences = [self.get_sentences(p) for p in paragraphs]
        all_sentences = [s for sublist in sentences for s in sublist]
        total_words = sum(len(self.tokenize(s)) for s in all_sentences)
        return total_words / len(all_sentences) if all_sentences else 0

    def average_word_length(self):
        """
        Calculate the average length of words in the text.
        """
        normalized_text = self.normalize_text()
        tokens = normalized_text.split()
        total_chars = sum(len(token) for token in tokens)
        return total_chars / len(tokens) if tokens else 0
    
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
        self._training_data_size = value

    
    def prepareTextInput(self,text,category):
        '''
        conversion of text into matrix X and vector y
        and fill up the list of variable as empty
        
        each sentence is considered as one trajectory.
        each word as a node
        the vector y is just repetition of string category into a vector
        '''
        sentences=self.text2Sentences(text)
        
        mR=len(sentences)
        self.variables=None
        y=[category]*mR
        X=[]
        for sentence in sentences:
            lstTokens = self.sentence2Tokens(sentence)
            X.append(lstTokens)
    
        return X,y

if __name__=='__main__':
    # import time
    # start_time = time.time()
    
    # dataFolder=r"C:\Users\Kardi\Documents\Kardi\Personal\Tutorial\NetworkScience\IdealFlow\Software\Python\Data Science\ReusableData\Text\\"
    # # fName="LookingBackward.txt"
    # fName="diary.txt"
    # f = open(dataFolder+fName,'r')
    # textInput=f.read()
    # f.close()
    
    # # textInput2="Aku mau makan nasi, bukan bubur madura. bukan bubur yang itu. bubur yang ini. Maka dari itu sebabnya."
    # # textInput="because the world never been never impossible."
    
    # category=fName[:-4]
    # print("download time: %s seconds ---" % (time.time() - start_time))
    # start_time = time.time()
    # tp=TextProcessing(markovOrder=2,name=category)
    # X,y=tp.prepareTextInput(textInput,category)
    # print("prepare input time: %s seconds ---" % (time.time() - start_time))
    # start_time = time.time()
    
    # print('accuracy = ',tp.ifnc.fit(X, y),'\n')
    
    # # X,y=tp.prepareTextInput(textInput2,"category2")
    # # print('accuracy = ',tp.ifnc.fit(X, y),'\n')
    # # lut=tp.ifnc.lut
    # # print('\nlut=',lut)
    # tp.ifnc.save()
    # # # ifnc.show()
    # print("training time: %s seconds ---" % (time.time() - start_time))
    # start_time = time.time()
    # # print('Sentence for',category,":\n")
    # # sentences=""
    # # for i in range(15):
    # #     tr=tp.ifnc.generate(category)
    # #     sentence=tp.tokens2Sentence(tr)
    # #     sentences=sentences+" "+ sentence
    # # print(sentences,"\n")
    # # print("generating time: %s seconds ---" % (time.time() - start_time))
    
    # tp=TextProcessing(markovOrder=2,name=category)
    # tp.ifnc.load()
    # sentences=""
    # for i in range(15):
    #     tr=tp.ifnc.generate(category)
    #     sentence=tp.tokens2Sentence(tr)
    #     sentences=sentences+" "+ sentence
    # print(sentences,"\n")
    
    # print(tp.ifnc.IFNs['diary'].nodesFlow())
    
    text = """
    Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence.
    It is concerned with the interactions between computers and human language.

    In particular, NLP is used to program computers to process and analyze large amounts of natural language data.
    """
    tp=TextProcessing()
    tp.text=text
        
    # Get paragraphs
    paragraphs = tp.get_paragraphs()
    print("Paragraphs:", paragraphs)

    # Get sentences from the first paragraph
    sentences = tp.get_sentences(paragraphs[0])
    print("Sentences:", sentences)

    # Tokenize the first sentence
    tokens = tp.tokenize(sentences[0])
    print("Tokens:", tokens)

    # Remove stop words
    filtered_tokens = tp.remove_stopwords(tokens)
    print("Filtered Tokens:", filtered_tokens)

    # Get bag of words
    bow = tp.bag_of_words(filtered_tokens)
    print("Bag of Words:", bow)

    # Generate bigrams
    bigrams = tp.ngrams(filtered_tokens, n=2)
    print("Bigrams:", bigrams)

    # Additional methods
    avg_sentence_length = tp.average_sentence_length()
    print("Average Sentence Length:", avg_sentence_length)

    vocab_size = tp.vocabulary_size
    print("Vocabulary Size:", vocab_size)

    # Predict intent and entities
    input_text = "Book a flight to New York on September 21st at 5:00 PM"
    prediction = tp.predict(input_text)
    print("Input Text:", input_text)
    print("Predicted Intent:", prediction['intent'])
    print("Identified Entities:", prediction['entities'])

    # Test data for evaluation
    test_data = [
        {"text": "Book a flight to Paris", "intent": "book_flight"},
        {"text": "What's the weather like today?", "intent": "weather_query"},
        {"text": "Hi there!", "intent": "greeting"},
        {"text": "Order a pizza for delivery", "intent": "order_food"}        
    ]

    # Evaluate the model
    metrics = tp.evaluate(test_data)
    print("Evaluation Metrics:", metrics)

    sentence = "John reads books."
    parsed = tp.parse_sentence(sentence)
    print("Parsed Sentence:", parsed)

    # Add a new intent
    tp.add_intent('schedule_meeting', ['schedule', 'meeting', 'appointment'])

    # Add a new entity pattern
    tp.add_entity_pattern('Email', r'\b[\w.-]+@[\w.-]+\.\w{2,4}\b')

    # Update model with new training data
    new_data = [
        {"text": "Schedule a meeting with Alice", "intent": "schedule_meeting"},
        {"text": "I have an appointment at 3 PM", "intent": "schedule_meeting"}
    ]
    tp.update_model(new_data)

    # Save patterns
    # asyncio.run(tp.save_patterns())
    tp.save_intent()

    # Predict intent and entities
    input_text = "Schedule a meeting with Bob on October 10th"
    prediction = tp.predict(input_text)
    print("Input Text:", input_text)
    print("Predicted Intent:", prediction['intent'])
    print("Identified Entities:", prediction['entities'])

    # Parse a sentence
    parsed = tp.parse_sentence("Alice sends an email.")
    print("Parsed Sentence:", parsed)
    