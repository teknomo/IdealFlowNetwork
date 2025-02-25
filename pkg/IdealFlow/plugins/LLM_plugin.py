# plugins/LLM_plugin.py
from plugin import PluginInterface
import requests

class LLMPlugin(PluginInterface):
    def __init__(self):
        # self.api_host = "http://localhost:11434/v1/chat/completions" # Ollama default
        self.api_host = "http://localhost:1234/v1/chat/completions" # LM Studio default
        # self.model = "deepseek-r1:1.5b" # Match your local model name
        self.model = "lmstudio-community/qwen2.5-7b-instruct"
        self.headers = {"Content-Type": "application/json"}
        self.api_key = "ollama" # api_key="lm-studio"
        self.chat_history = []
        self.max_tokens = 32000
        self.max_history_length = 1000
        self.temperature = 0.7
  
    def get_actions(self):
        return {
            'setup_LLM': self.setupLLM,
            'chat_once': self.chatOnce,            
            'chat_continuous': self.chat_continuous,
            'chat_deep': self.chat_deep 
        }

    def setupLLM(self, api_host, api_key, model, **kwargs):
        if api_host != "":
            self.api_host = api_host
        if api_key != "":
            self.api_key = api_key
        if model != "":
            self.model = model

    def chatOnce(self, userPrompt, systemPrompt, isDeepThinking=False, **kwargs):
        try:
            # Construct the messages list with system prompt and chat history
            messages = [{"role": "system", "content": systemPrompt}] + self.chat_history + [{"role": "user", "content": userPrompt}] 
            data = {
                "model": self.model,
                "messages":  messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            response = requests.post(self.api_host, headers=self.headers, json=data)
            if response.status_code==200:
                if isDeepThinking:
                    res = response.json()
                    content = res['choices'][0]['message']['content'] # get content
                    return content
                else:
                    return self.response2content(response.json())
            else:
                print ('response:',response)
        except Exception as e:
            print(f"\nAn error occurred in chatOnce: {str(e)}")
                
    def response2content(self,response):
        # remove <think></think> if exists
        try:
            content = response['choices'][0]['message']['content'] # get content
            think_index = content.find('<think>')   # index of the <think> tag
            if think_index != -1: # Slicing the content after <think> tag
                # Adding the length of the <think> tag to get the content after it
                extracted_content = content[think_index + len('<think>'):].strip()
                end_think_index = extracted_content.find('</think>')
                if end_think_index != -1:
                    extracted_content = extracted_content[end_think_index + len('</think>'):].strip()
                return extracted_content
            else:
                return content # No <think> tag found.
        except Exception as e:
            print(f"\nAn error occurred in response2content: {str(e)}")
            return response
    
    def chat_continuous(self, systemPrompt="", **kwargs):
        self.chat(systemPrompt,False, **kwargs)
    
    def chat_deep(self, systemPrompt="", **kwargs):
        self.chat(systemPrompt,True, **kwargs)
        
    def chat(self, systemPrompt="", isDeepThinking=False, **kwargs):
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()

            # Check for quit command
            if user_input.lower() == "exit" or user_input.lower() == "bye" or user_input.lower() == "quit":
                print("Agent: Goodbye!")
                break
            
            try:
                response = self.chatOnce(user_input, systemPrompt, isDeepThinking)
                print("\nAgent:", response)
                
                # Add the user input to the chat history
                self.chat_history.append({"role": "user", "content": user_input})
                # manage history to avoid exceeding max token limit
                if len(self.chat_history) > self.max_history_length:
                    self.chat_history = self.chat_history[-self.max_history_length:]

            except Exception as e:
                print(f"\nAn error occurred in chat: {str(e)}")
                break
        