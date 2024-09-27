import streamlit as st
from groq import Groq
from pymongo import MongoClient
import bcrypt
from topic_feedback import extract_topic
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# Function to extract conversation history
def extract_conversation_history(username, db):
    collection = db["conversations"]
    user_conversations = collection.find_one({"username": username})
    if user_conversations and "topics" in user_conversations:
        # Flattening user responses from all topics
        user_responses = []
        for topic in user_conversations["topics"]:
            user_responses.extend(topic["conversation"].splitlines())
        return user_responses
    else:
        return []

# Function to calculate a critical thinking score based on the user's response
def calculate_critical_thinking_score(user_input):
    score = 0
    # Criteria 1: Length of response
    if len(user_input) > 50:
        score += 1  # Add 1 point for responses longer than 50 characters
    
    # Criteria 2: Presence of reasoning keywords
    reasoning_keywords = ["because", "therefore", "so", "thus", "hence"]
    if any(keyword in user_input.lower() for keyword in reasoning_keywords):
        score += 2  # Add 2 points if reasoning is present
    
    # Criteria 3: Clarifying questions
    if "?" in user_input:
        score += 1  # Add 1 point if the user asks clarifying questions
    
    # Criteria 4: Use of problem-solving keywords
    problem_solving_keywords = ["solve", "solution", "fix", "approach", "resolve"]
    if any(keyword in user_input.lower() for keyword in problem_solving_keywords):
        score += 2  # Add 2 points if user attempts problem-solving
    
    return score

# Function to analyze critical thinking across the conversation
def analyze_critical_thinking(conversations):
    critical_thinking_scores = []
    
    # Analyze each user's response
    for conversation in conversations:
        conversation_lines = conversation.splitlines()
        for line in conversation_lines:
            if line.startswith("User:"):
                user_response = line[len("User: "):].strip()  # Extract user response
                score = calculate_critical_thinking_score(user_response)
                critical_thinking_scores.append(score)
    
    return critical_thinking_scores

# Function to plot critical thinking over time (line plot)
def plot_critical_thinking(scores):
    fig, ax = plt.subplots()
    
    # Create the line plot
    ax.plot(range(len(scores)), scores, marker='o', linestyle='-', color='b', label='Critical Thinking Score')
    
    # Add labels and title
    ax.set_xlabel('Conversation Turn (User Responses)')
    ax.set_ylabel('Critical Thinking Score')
    ax.set_title('Critical Thinking Progression Over Conversation')
    
    # Show grid for better readability
    ax.grid(True)
    
    # Display the plot
    st.pyplot(fig)

# Hardcoded credentials
MONGODB_URL = "mongodb+srv://nishantdesale1402:t6sOHW09rUyp9PLR@socraticai.iryyu.mongodb.net/?retryWrites=true&w=majority&appName=SocraticAI&tls=true&tlsAllowInvalidCertificates=true"
GROQ_API_KEY = "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"

# Function to initialize the Groq client
def initialize_groq_client(api_key):
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Groq client: {e}")
        return None

# Function to score user interactions (basic scoring)
def calculate_score(user_input, assistant_response):
    score = 0
    if "correct" in assistant_response.lower():
        score += 10  # Award points for correct answers
    if len(user_input) > 50:
        score += 5  # Reward longer and more thoughtful responses
    return score

# Function to update user score in the database
def update_score(username, score, db):
    users_collection = db["users"]
    user = users_collection.find_one({"username": username})
    if user:
        current_score = user.get("score", 0)
        new_score = current_score + score
        users_collection.update_one({"username": username}, {"$set": {"score": new_score}})
        return new_score
    return None

def socratic_assistant_response(client, input_text, context=None, prompt=None):
    guidelines = """
    If a code is provided identify the issues or the changes to be made. and revolve around the issue.
    1. Focus on Needs: Address the student’s current understanding and direct questions around their needs.
    2. Concise Reviews: Keep reviews of correct answers brief and to the point.
    3. One or Two Questions: Limit your questions to one or two at a time to avoid overwhelming the student.
    4. Efficient Conclusions: Aim to reach conclusions effectively and quickly, covering all necessary aspects.
    5. Varied Question Types: Utilize different question formats, including MCQs and fill-in-the-blanks, to enhance learning.
    6. Avoid Long Answers: Steer clear of questions that require lengthy responses.
    """
    
    # System prompt
    if prompt:
        system_prompt = prompt
    else:
        system_prompt = f"""
        You are a Socratic teaching assistant. Your role is to guide students through learning by asking probing questions rather than providing direct answers. Help the student discover the correct answers by themselves, leading them step-by-step.
        Here are guidelines you must adhere to: {guidelines}

        Question Formats:

        Generate an open-ended question with the following details:
        - **Question:** [Insert your open-ended question here]

        Generate a multiple-choice question with the following details:
        - **Question:** [Insert your question here]
        - **Options:**
          - **A:** [Option A]
          - **B:** [Option B]
          - **C:** [Option C]
          - **D:** [Option D]

        Generate a fill-in-the-blanks question with the following details:
        - **Sentence:** [Insert sentence with a blank here]
        """

    conversation = f"{context}\nStudent: {input_text}\nAssistant:" if context else f"Student: {input_text}\nAssistant:"
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation}
            ],
            model="llama3-70b-8192",
            temperature=0.5
        )
        response = chat_completion.choices[0].message.content
        return response
    except Exception as e:
        st.error(f"Error generating chat completion: {e}")
        return "An error occurred while generating the response."

def store_conversation(username, conversation_str, db):
    collection = db["conversations"]

    # Extract topic
    topic = extract_topic(conversation_str)

    existing_user = collection.find_one({"username": username})

    if existing_user:
        # Check if topic exists
        topics = existing_user.get("topics", [])
        topic_exists = False
        for t in topics:
            if t.get("topic") == topic:
                # Update existing conversation
                collection.update_one(
                    {"username": username, "topics.topic": topic},
                    {
                        "$set": {
                            "topics.$.conversation": conversation_str
                        }
                    }
                )
                topic_exists = True
                break
        if not topic_exists:
            # Add new topic with conversation
            collection.update_one(
                {"username": username},
                {
                    "$push": {
                        "topics": {
                            "topic": topic,
                            "conversation": conversation_str
                        }
                    }
                }
            )
    else:
        # Insert new user and topic
        conversation_data = {
            "username": username,
            "topics": [
                {
                    "topic": topic,
                    "conversation": conversation_str
                }
            ]
        }
        collection.insert_one(conversation_data)

# Streamlit app
def main():
    st.set_page_config(page_title="Socratic Teaching Assistant", page_icon=":books:")

    st.sidebar.title("User Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if not username or not password:
        st.sidebar.info("Please enter your username and password to continue.")
        st.stop()

    # Initialize MongoDB client
    try:
        client_mongo = MongoClient(MONGODB_URL)
        db = client_mongo["SocraticAI"]
        users_collection = db["users"]

        conversations_collection = db["conversations"]
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {e}")
        st.stop()

    # Fetch stored password hash from the database
    user = users_collection.find_one({"username": username})

    # Hardcoded password hash for "123"
    hardcoded_password_hash = bcrypt.hashpw("123".encode('utf-8'), bcrypt.gensalt())

    if user is None:
        # Register new user
        users_collection.insert_one({
            "username": username,
            "password_hash": hardcoded_password_hash.decode('utf-8'),
            "score": 0  # Initialize score
        })
        st.sidebar.success(f"New user registered: {username}")
    else:
        stored_password_hash = user.get("password_hash")
        if not stored_password_hash:
            st.sidebar.error("Password not set for this user.")
            st.stop()

        # Check password
        if not bcrypt.checkpw(password.encode('utf-8'), stored_password_hash.encode('utf-8')):
            st.sidebar.error("Invalid username or password.")
            st.stop()

    st.title("📚 Gamified Learning Management System (LME)")

    # Initialize Groq client
    client_groq = initialize_groq_client(GROQ_API_KEY)
    if client_groq is None:
        st.error("Failed to initialize the Groq client. Please check your API key.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you learn today?"}]

    if "selected_topic" not in st.session_state:
        st.session_state["selected_topic"] = None

    if "expander_open" not in st.session_state:
        st.session_state["expander_open"] = False

    # Handle user input
    user_input = st.chat_input("Enter your question or response:")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Prepare context
        context = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages])

        # Generate assistant response
        try:
            full_response = socratic_assistant_response(client_groq, user_input, context=context)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # Calculate score based on user interaction
            score = calculate_score(user_input, full_response)
            new_score = update_score(username, score, db)
            st.sidebar.success(f"Your score is now: {new_score}")

        except Exception as e:
            st.error(f"An error occurred while generating the response: {e}")

    # Display chat messages
    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["content"])
        elif msg["role"] == "user":
            st.chat_message("user").write(msg["content"])

    if st.button("Analyze Critical Thinking"):
    # Extract conversation history from the database
        conversations = extract_conversation_history(username, db)
        if conversations:
            # Analyze critical thinking scores
            critical_thinking_scores = analyze_critical_thinking(conversations)
            
            # Check if there are scores to plot
            if critical_thinking_scores:
                st.write("Critical Thinking Scores Over Time")
                plot_critical_thinking(critical_thinking_scores)
            else:
                st.write("No user responses found for analysis.")
    else:
        st.write("No conversation history available.")

    # Sidebar for displaying history
    with st.sidebar:
        st.header("Conversation History & Rewards")
        # Show score
        user = users_collection.find_one({"username": username})
        if user and "score" in user:
            st.write(f"**Your Score:** {user['score']}")
            if user["score"] > 50:
                st.write("🎉 **Reward:** Gold Badge")
            elif user["score"] > 20:
                st.write("🎉 **Reward:** Silver Badge")
            else:
                st.write("🎉 **Reward:** Bronze Badge")

        if st.button("Show History"):
            st.session_state["expander_open"] = not st.session_state["expander_open"]

        with st.expander("Conversation History", expanded=st.session_state["expander_open"]):
            # Fetch topics from the database for the user
            user_conversations = conversations_collection.find_one({"username": username})
            if user_conversations and "topics" in user_conversations:
                topics = [t["topic"] for t in user_conversations["topics"]]
                selected_topic = st.selectbox("Select Topic", options=topics, index=None , placeholder="Select Chat" , key="topic_selector")

                st.session_state["selected_topic"] = selected_topic

                if selected_topic:
                    # Find and display the conversation for the selected topic
                    for topic in user_conversations["topics"]:
                        if topic["topic"] == selected_topic:
                            conversation_str = topic["conversation"]

                            # Display messages for the selected topic
                            conversation_lines = conversation_str.splitlines()
                            for line in conversation_lines:
                                if line.startswith("Assistant:"):
                                    st.markdown(f'<p style="color:yellow;">{line}</p>', unsafe_allow_html=True)
                                elif line.startswith("User:"):
                                    st.markdown(f'<p style="color:white;">{line}</p>', unsafe_allow_html=True)
            else:
                st.write("No conversation history available.")

        # Button for clearing the conversation
        if st.button("Clear Conversation"):
            # Process and store the current conversation before clearing
            if st.session_state["messages"]:
                conversation_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages])
                store_conversation(username, conversation_str, db)
            
            # Clear conversation
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you learn today?"}]

if __name__ == "__main__":
    main()
