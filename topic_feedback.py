import os
from groq import Groq

# Initialize Groq client
client = Groq(api_key="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")

def get_groq_response(prompt, model="llama3-8b-8192"):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt},
            ],
            model=model,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error fetching response from Groq API: {e}")
        return None

def extract_topic(conversation_text):
    # Generate a prompt to identify the main topic of the conversation
    prompt = (
        f"""Analyze the following conversation and respond with only the main topic in lesst than 4 words:
        Conversation: {conversation_text}"""

    )
    topic = get_groq_response(prompt)
    return topic

def generate_performance_feedback(conversation_text, topic):
    system_prompt = (
        f"Evaluate the user's performance on the following topic based the given conversation and topic"
        f"Conversation: {conversation_text}\n\n"
        f"Topic: {topic}\n\n"
        f"Provide detailed feedback on the user's performance and understanding of the topic:"
    )
    prompt = (f"Conversation: {conversation_text}\n\n"
        f"Topic: {topic}\n\n")
    # Generate a prompt to get feedback on the user's performance regarding the topic
    try:
        model="llama3-8b-8192"
        chat_completion = client.chat.completions.create(
            
            messages=[
                {"role": "user", "content": prompt},
                {"role": "System", "content": system_prompt}
            ],
            model=model,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error fetching response from Groq API: {e}")
        return None

def main():
    # Input conversation from the user
    user_conversation = input("Enter the conversation: ")
    
    # Extract the topic from the conversation
    topic = extract_topic(user_conversation)
    if topic is None:
        print("Failed to extract topic.")
        return
    
    # Generate performance feedback based on the extracted topic
    performance_feedback = generate_performance_feedback(user_conversation, topic)
    if performance_feedback is None:
        print("Failed to generate performance feedback.")
        return
    
    print(f"Conversation: {user_conversation}")
    print(f"Extracted Topic: {topic}")
    print(f"Performance Feedback: {performance_feedback}")

if __name__ == "__main__":
    main()
