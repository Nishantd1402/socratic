import streamlit as st
from groq import Groq
from pymongo import MongoClient
import bcrypt
from topic_feedback import extract_topic
import app
import streamlit.components.v1 as components
import google.generativeai as genai
from pdf import parse_pdf
import os
import yt_dlp


def download_youtube_audio(youtube_url, output_path='.', audio_format='mp3'):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': audio_format,
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_path, 'sample'),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=True)
        audio_file = os.path.join(output_path, "sample.mp3")

    return audio_file

def transcribe_audio(api_key, filename, model="distil-whisper-large-v3-en", prompt=None, language="en", temperature=0.1):
    client = Groq(api_key=api_key)

    with open(filename, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(filename, file.read()),
            model=model,
            prompt=prompt,
            response_format="json",
            language=language,
            temperature=temperature
        )

    return transcription.text

def split_audio(filename, chunk_length=60):
    """
    Splits the audio file into chunks of specified length (in seconds).
    """
    from pydub import AudioSegment
    
    audio = AudioSegment.from_file(filename)
    chunks = []
    
    for i in range(0, len(audio), chunk_length * 1000):  # Convert seconds to milliseconds
        chunk = audio[i:i + chunk_length * 1000]
        chunk_filename = f"{filename}_chunk{i // 1000}.mp3"
        chunk.export(chunk_filename, format="mp3")
        chunks.append(chunk_filename)
    
    return chunks



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
    
def initialize_gemini(key):
    genai.configure(api_key=key)

def get_gemini_answers(input_text, context=None, prompt=None , pdf=None):
    try:
        initialize_gemini("AIzaSyAalmVnZp-_w2YD7KXDR2iG5xtxlx7LE_U")
    except Exception as e:
        st.error(e)
    guidelines = """
    If a code is provided identify the issues or the changes to be made. and revolve around the issue.
    1. Focus on Needs: Address the student’s current understanding and direct questions around their needs.
    2. Concise Reviews: Keep reviews of correct answers brief and to the point.
    3. One or Two Questions: Limit your questions to one or two at a time to avoid overwhelming the student.
    4. Efficient Conclusions: Aim to reach conclusions effectively and quickly, covering all necessary aspects.
    5. Varied Question Types: Utilize different question formats, including MCQs and fill-in-the-blanks, to enhance learning.
    6. Avoid Long Answers: Steer clear of questions that require lengthy responses.
    """
    
    if pdf:
        pdf = pdf
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
                {"role": "user", "parts": conversation}])
    response = chat_session.send_message(input_text)
    response_text = response.text if hasattr(response, 'text') else str(response)

    return response_text


import streamlit as st

def chat_with_pdf_text(input_text, pdf_text , format):
    try:
        initialize_gemini("AIzaSyAalmVnZp-_w2YD7KXDR2iG5xtxlx7LE_U")
    except Exception as e:
        st.error(e)

    # System prompt with the user's specified prompt
    system_prompt = (
        f"You are an intelligent assistant designed to help users understand and analyze  text provided by {format}. "
        "When a user asks a question, consider the content of the PDF document to provide relevant and "
        "accurate answers. If the question cannot be answered based on the content, indicate that you don't "
        "have sufficient information. Always be polite and helpful in your responses."
    )

    # Conversation context
    conversation = f"PDF Content: {pdf_text}\nStudent: {input_text}\nAssistant:"

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
        system_instruction=system_prompt
    )

    chat_session = model.start_chat(history=[
        {"role": "model", "parts": f"{system_prompt}"},
        {"role": "user", "parts": conversation}
    ])
    response = chat_session.send_message(input_text)
    response_text = response.text if hasattr(response, 'text') else str(response)

    return response_text


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

def get_user_scores_and_rank(db):
    """
    Retrieve all users and their scores from the MongoDB 'user' collection, rank them based on their score.
    
    Parameters:
    - db: MongoDB database connection object
    
    Returns:
    - A list of tuples with user rankings in the format [(rank, username, score), ...]
    """
    try:
        user_collection = db['users']
        users = list(user_collection.find({}, {"username": 1, "score": 1}).sort("score", -1))

        # Check if any users are found
        if not users:
            print("No users found in the 'user' collection.")
            return []

        # Rank users based on their score
        ranked_users = []
        rank = 1
        for user in users:
            username = user.get("username", "Unknown")
            score = user.get("score", 0)
            ranked_users.append((rank, username, score))
            rank += 1

        return ranked_users

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


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

    st.title("📚 MVP - Your AI Learning Buddy")

    # Initialize Groq client
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Socratic-AI", "Leaderboard" , "PDF - Mate" , "YT - Genie" , "Chat History and MindMap"])
    with tab1:
    # Initialize session state variables if they don't exist
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you learn today?"}]
        if "selected_topic" not in st.session_state:
            st.session_state["selected_topic"] = None
        if "expander_open" not in st.session_state:
            st.session_state["expander_open"] = False

        

        # Handle user input
        user_input = st.chat_input("Enter your question or response:" , key = "Socratic")

        if user_input:
            # Append user message to the session state
            st.session_state["messages"].append({"role": "user", "content": user_input})

            # Prepare context using the updated messages
            context = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state["messages"]])

            try:
                # Generate assistant response based on user input and context
                full_response = get_gemini_answers(user_input, context=context)
                
                # Append assistant response to the session state
                st.session_state["messages"].append({"role": "assistant", "content": full_response})


                # Calculate score based on user interaction
                score = calculate_score(user_input, full_response)
                new_score = update_score(username, score, db)
                st.sidebar.success(f"Your score is now: {new_score}")

            except Exception as e:
                st.error(f"An error occurred while generating the response: {e}")
            
            # Display chat messages
        for msg in st.session_state["messages"]:
            if msg["role"] == "assistant":
                st.chat_message("assistant").write(msg["content"])
            elif msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            
                

        

    with tab2:
        # Retrieve ranked users and display them
        ranked_users = get_user_scores_and_rank(db)
        
        if ranked_users:
            st.write("Here is the current ranking of users:")
            
            # Create a leaderboard table with rank, username, and score
            leaderboard_data = {
                "Rank": [rank for rank, _, _ in ranked_users],
                "Username": [username for _, username, _ in ranked_users],
                "Score": [score for _, _, score in ranked_users]
            }

            # Display the leaderboard using Streamlit's dataframe feature
            st.dataframe(leaderboard_data , width = 600)
    
    with tab3:
        with st.container():
            # File uploader for a PDF
            uploaded_file = st.file_uploader("Choose a PDF file", accept_multiple_files=False)

            # Parse the PDF if a file is uploaded
            if uploaded_file is not None:
                content, img = parse_pdf(uploaded_file)  # Parse the PDF into content and image



                # Initialize chat history in session state for the PDF-based chat
                if "chatpdf" not in st.session_state:
                    st.session_state["chatpdf"] = [{"role": "assistant", "content": "How can I help you with this document?"}]
                    
                
                # User input field for asking questions
                user_input = st.chat_input("Ask a question about the PDF:", key="pdf")

                

                # If there's user input, process it
                if user_input:
                    # Append user message to the chat history
                    st.session_state["chatpdf"].append({"role": "user", "content": user_input})

                    try:
                        # Generate assistant's response using the user's input and PDF content
                        full_response = chat_with_pdf_text(user_input, content , format = "PDF")

                        # Append assistant response to the chat history
                        st.session_state["chatpdf"].append({"role": "assistant", "content": full_response})

                    except Exception as e:
                        st.error(f"An error occurred while generating the response: {e}")

                if st.session_state["chatpdf"]:  # Check if chatpdf is not empty
                    for chat in st.session_state["chatpdf"]:
                        if chat["role"] == "assistant":
                            st.chat_message("assistant").write(chat["content"])
                        elif chat["role"] == "user":
                            st.chat_message("user").write(chat["content"])

                

            else:
                st.write("Please upload a PDF to start interacting.")

    with tab4:
        link = st.text_input("YouTube Link")
        api_key = "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"

        if link and "transcription" not in st.session_state:
            # Download and process the audio only if it's not already in session_state
            audio_file_path = download_youtube_audio(link)

            # Split audio into chunks of 60 seconds
            audio_chunks = split_audio(audio_file_path)

            full_transcription = ""
            for chunk in audio_chunks:
                transcription_text = transcribe_audio(api_key, chunk, prompt="Transcribe correctly without spelling mistakes.")
                full_transcription += transcription_text + "\n"
                os.remove(chunk)
            
            # Store the transcription in session_state to avoid reprocessing
            st.session_state["transcription"] = full_transcription

        # Check if transcription is available in session_state
        if "transcription" in st.session_state:
            full_transcription = st.session_state["transcription"]

            if "chattube" not in st.session_state:
                st.session_state["chattube"] = [{"role": "assistant", "content": "How can I help you with this video?"}]
            
            # User input field for asking questions
            user_input = st.chat_input("Ask a question about the video:", key="youtube")

            # If there's user input, process it
            if user_input:
                # Append user message to the chat history
                st.session_state["chattube"].append({"role": "user", "content": user_input})

                try:
                    # Generate assistant's response using the user's input and the video transcription
                    full_response = chat_with_pdf_text(user_input, full_transcription , format = "Youtube")

                    # Append assistant response to the chat history
                    st.session_state["chattube"].append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error(f"An error occurred while generating the response: {e}")

            # Display the chat history
            if st.session_state["chattube"]:  # Check if chat history is not empty
                for chat in st.session_state["chattube"]:
                    if chat["role"] == "assistant":
                        st.chat_message("assistant").write(chat["content"])
                    elif chat["role"] == "user":
                        st.chat_message("user").write(chat["content"])



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

        # Button for clearing the conversation
        if st.button("Clear Conversation"):
            # Process and store the current conversation before clearing
            if st.session_state["messages"]:
                conversation_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages])
                store_conversation(username, conversation_str, db)
            
            # Clear conversation
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you learn today?"}]

    with tab5:
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
                                    st.markdown(f'<p style="color:black;">{line}</p>', unsafe_allow_html=True)
                                elif line.startswith("User:"):
                                    st.markdown(f'<p style="color:black;">{line}</p>', unsafe_allow_html=True)
                            
                            app.create_map(conversation_str)
                            with open("mark.html", "r", encoding="utf-8") as file:
                                html_content = file.read()

                            # Embed the HTML content in Streamlit
                            components.html(html_content , height=400)


            else:
                st.write("No conversation history available.")

        

        


if __name__ == "__main__":
    main()
