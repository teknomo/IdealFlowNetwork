# example_chatbot.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../IdealFlow")))
from Automation import Automation

if __name__=='__main__':
    # run Ollama API or LM Studio before running this code 
    
    server = "ollama" # "ollama" # "lms"
    system_prompt = 'You are a useful assistant. I am still a child in elementary school. Can you explain everything in step by step and in easy and clear way?'

    if server=="ollama":
        # Before calling this code, run Ollama server with certain model in CLI
        # > ollama serve
        api_host = "http://localhost:11434/v1/chat/completions" # Ollama default
        model = "deepseek-r1:1.5b" # "deepseek-r1:latest"    # "deepseek-r1:1.5b"      # Match your local model name
        api_key = "ollama" 
    elif server == "lms":
        # Before calling this code, run LM Studio server in CLI
        # > lms server start
        # > lms load
        api_host = "http://localhost:1234/v1/chat/completions" # LM Studio default
        api_key = "lm-studio"
        model = "lmstudio-community/qwen2.5-7b-instruct"
        # model = "MiniCPM-o-2_6-gguf/Model-7.6B-Q4_0.gguf"
        
    
    af = Automation("Chatbot")
    af.add_node("a", action='setup_LLM', params={'api_host': api_host, 'api_key': api_key, 'model': model})
    af.add_node("b", action='chat_continuous', params={'systemPrompt': system_prompt})
    # af.add_node("c", action='chat_deep', params={'systemPrompt': system_prompt})
    
    # Define dependencies
    af.assign(["a", "b"])
    # af.assign(["a", "c"])

    # Save the automation for future loading
    # af.save(file_path)

    # Execute the automation
    af.execute()

    # Show the IFN structure
    # af.show()
    
    