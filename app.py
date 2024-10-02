import subprocess
import streamlit as st

import os

import streamlit.components.v1 as components

import os
from groq import Groq
import google.generativeai as genai






def create_map(text):
    genai.configure(api_key="AIzaSyBF8WYy0-xa4rVjYesBfSzYFJbNgpQnXMA")
    
    def create_markdown_file(pre_written_text, user_text, output_file_path):
        combined_text = pre_written_text + "\n\n" + user_text
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(combined_text)

    pre_written_text = """---    
    markmap:
        colorFreezeLevel: 2
        maxWidth: 200
        embedAssets: true
        initialExpandLevel: 3
    ---
        """

    system_prompt = "You will be given an input of a conversation. analyze it to provide a markdown code which grasps all of it's concepts and help map the whole conversation.\
                Give only markdown code without any other text with using only '#' symbol and no indentation at the start of each line for creating a mind map. The mind map should visually represent the relationships and hierarchy of all the complete information in the given input.\
                Use meaningful connections between the concepts and organize them logically.\
                Instructions:\
                    1. Give only markdown code without any other text. Use only '#' symbol. No indentation at the start of each line.\
                    2. Use all the concepts from the inputs and make it lengthy.\
                    3. Identify the main topics and subtopics.\
                    4. Create a dense lengthy hierarchical relationships between these topics.\
                    6. Explain the concepts of the text for clear understanding and be authentic to the given input.\
                Example text : Artificial Intelligence (AI) is a branch of computer science. It includes machine learning, where computers learn from data.\
                    Deep learning is a subset of machine learning involving neural networks. AI is used in various fields such as healthcare, finance, and transportation.\
                    In healthcare, AI helps in diagnosing diseases. In finance, AI is used for fraud detection.\
                    In transportation, AI powers autonomous vehicles.\
                output:\
                # Mind Map\
                ## Artificial Intelligence (AI)\
                ### Branches\
                #### Machine Learning\
                ##### Computers learn from data\
                #### Deep Learning\
                ##### Subset of machine learning\
                ##### Involves neural networks\
                ### Applications\
                #### Healthcare\
                ##### Diagnosing diseases\
                #### Finance\
                ##### Fraud detection\
                #### Transportation\
                ##### Autonomous vehicles"
    
    generation_config = {
    "temperature": 0.35,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    # safety_settings = Adjust safety settings
    # See https://ai.google.dev/gemini-api/docs/safety-settings
    system_instruction=system_prompt)

    chat_session = model.start_chat(history=[{"role": "model", "parts": f"system_prompt"},
                ])
    response = chat_session.send_message(text)
    response_text = response.text if hasattr(response, 'text') else str(response)

    file_name = f"mark.md"
    html_file = f"mark.html"  # Store in the html_folder

    # Create markdown file
    create_markdown_file(pre_written_text, response_text, file_name)

    # Convert markdown to HTML using markmap
    command = f'npx markmap-cli {file_name} --no-open -o {html_file}'
    subprocess.run(command, shell=True)

    return response_text


        