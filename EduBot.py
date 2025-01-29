import streamlit as st
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from pdfminer.high_level import extract_text as extract_text_pdfminer
import fitz  # PyMuPDF (for fallback extraction)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from collections import Counter
from nltk.corpus import stopwords
from nltk import pos_tag
import nltk
import wikipedia
from keybert import KeyBERT

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

kw_model = KeyBERT(model='distilbert-base-nli-mean-tokens')



def extract_key_phrases(text, num_phrases=5):
    """ Extract key phrases from text using KeyBERT """
    key_phrases = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', use_maxsum=True, nr_candidates=20, top_n=num_phrases)
    return [phrase[0] for phrase in key_phrases]

def generate_question_answer_pairs(text):
    """ Generate question-answer pairs based on key phrases """
    key_phrases = extract_key_phrases(text)
    qa_pairs = []

    for phrase in key_phrases:
        # Here you can format your questions based on key phrases
        question = f"What can you tell about {phrase}?"
        answer = phrase  # For simplicity, the answer is the key phrase itself
        qa_pairs.append((question, answer))

    return qa_pairs

def evaluate_answer(user_answer, correct_answer, threshold=0.7):
    """ Evaluate the user's answer with a simple heuristic """
    correct_tokens = set(correct_answer.lower().split())
    user_tokens = set(user_answer.lower().split())
    common_tokens = user_tokens.intersection(correct_tokens)
    score = len(common_tokens) / len(correct_tokens)
    return round((score / threshold) * 5)  # normalize and convert to a score out of 5




def get_topics_from_text(text, num_topics=5):
    # Tokenize and filter out stopwords and non-alphanumeric words
    words = [word.lower() for word in word_tokenize(text) if word.isalnum()]
    words = [word for word in words if word not in stopwords.words('english')]
    
    # POS tagging and filtering nouns
    nouns = [word for word, pos in pos_tag(words) if pos.startswith('NN')]
    
    # Get most common nouns as topics
    most_common_nouns = [word for word, count in Counter(nouns).most_common(num_topics)]
    
    return most_common_nouns

def generate_questions(text, topics, questions_per_topic=2):
    sentences = sent_tokenize(text)
    questions = []

    for topic in topics:
        # Filter sentences that contain the topic
        topic_sentences = [sent for sent in sentences if topic in sent.lower()]
        
        for _ in range(questions_per_topic):
            if topic_sentences:
                sentence = random.choice(topic_sentences)
                question_text = sentence.replace(topic, "_____", 1)
                
                # Create dummy options
                options = [topic] + random.sample(topics, 3)
                random.shuffle(options)

                questions.append({
                    "text": question_text, 
                    "options": options, 
                    "answer": topic,
                    "topic": topic  # Using the topic itself for feedback
                })
    
    return questions

def generate_quiz_from_text(text, num_questions=5):
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    all_words = set(word_tokenize(text))
    questions = []

    for _ in range(num_questions):
        # Randomly select a sentence to base the question on
        sentence = random.choice(sentences)
        words = word_tokenize(sentence)

        # Randomly select a word from the sentence as the answer
        answer = random.choice(words)
        while not answer.isalnum() or len(answer) < 4:  # Ensure the answer is a significant word
            answer = random.choice(words)

        # Create a list of options, including the answer
        options = [answer]
        while len(options) < 4:
            distractor = random.choice(list(all_words))
            if distractor not in options and distractor.isalnum() and len(distractor) >= 4:
                options.append(distractor)
        random.shuffle(options)  # Shuffle the options to randomize the answer's position

        # Format the question by replacing the answer with a blank
        question_text = sentence.replace(answer, "_____", 1)
        questions.append({"text": question_text, "options": options, "answer": answer})

    return questions




def display_quiz(questions):
    user_answers = []
    incorrect_topics = set()
    for i, question in enumerate(questions):
        st.write(f"Q{i+1}: {question['text']}")
        user_answer = st.radio("Choose the correct answer:", question["options"], key=f"question_{i}")
        user_answers.append(user_answer)
        if user_answer != question["answer"]:
            incorrect_topics.add(question['topic'])

    submit = st.button("Submit Quiz")
    if submit:
        correct_count = sum(user_answer == question["answer"] for user_answer, question in zip(user_answers, questions))
        st.write(f"You got {correct_count}/{len(questions)} correct.")

        st.subheader("Review Answers and Feedback:")
        for i, (user_answer, question) in enumerate(zip(user_answers, questions)):
            correct = user_answer == question["answer"]
            st.write(f"Q{i+1}: {question['text'].replace('_____', '**' + question['answer'] + '**')}")
            st.write(f"Your answer: {user_answer} - {'Correct' if correct else 'Incorrect'}")

        if incorrect_topics:
            st.subheader("Need More Practice? Here are some notes:")
            for topic in incorrect_topics:
                st.write(f"#### Notes for {topic}")
                # Display notes from the PDF content
                related_text = get_related_text_from_pdf(st.session_state.raw_text, topic)
                if related_text:
                    st.write(related_text)
                else:
                    st.write(f"No detailed notes found in the PDF for {topic}.")

                # Try to fetch a summary from Wikipedia
                try:
                    summary = wikipedia.summary(topic, sentences=3)
                    st.write(f"#### Summary from Wikipedia on {topic}:")
                    st.write(summary)
                except wikipedia.exceptions.DisambiguationError as e:
                    st.write(f"Multiple entries found for {topic}. Please be more specific.")
                except wikipedia.exceptions.PageError:
                    st.write(f"No Wikipedia page found for {topic}.")

def get_related_text_from_pdf(text, topic):
    sentences = sent_tokenize(text)
    related_sentences = [sentence for sentence in sentences if topic.lower() in sentence.lower()]
    return "\n".join(related_sentences[:5])  # Return the first 5 related sentences for brevity




def get_pdf_text(pdf_docs):
    text = ""
    code_blocks = []  # List to hold code blocks
    for pdf in pdf_docs:
        # Try PDFMiner first
        try:
            text += extract_text_pdfminer(pdf)
        except Exception as e:
            print(f"PDFMiner failed: {e}, trying PyMuPDF")
            # Fallback to PyMuPDF if PDFMiner fails
            try:
                doc = fitz.open("pdf", pdf.read())
                for page in doc:
                    text += page.get_text()
            except Exception as e:
                print(f"PyMuPDF failed: {e}")

    # Your existing code logic to identify and separate code blocks
    lines = text.split('\n')
    for line in lines:
        if "{" in line or "}" in line or line.strip().startswith("def "):  # Simple heuristic to identify code
            code_blocks.append(line)
        else:
            text += line + '\n'
    code_text = '\n'.join(code_blocks)
    return text, code_text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, store_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(store_name)

def process_documents(raw_text, code_text):
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks, "text_faiss_index")
    
    if code_text.strip():  # Process code text if present
        code_chunks = get_text_chunks(code_text)
        get_vector_store(code_chunks, "code_faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,
    if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain



def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        if "code" in user_question or "function" in user_question:  # Decide which vector store to use
            vector_store_name = "code_faiss_index"
        else:
            vector_store_name = "text_faiss_index"
        
        new_db = FAISS.load_local(vector_store_name, embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        
        # Using invoke as per the updated method call
        response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply: ", response["output_text"])

    except Exception as e:  # Catching a broader exception
        st.error("Unable to process the question. Please try rephrasing your question or try again later.")



def main():
    st.set_page_config("🤖EduBot")
    st.header("Welcome to EduBot🤖")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, type=['pdf'])
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text, code_text = get_pdf_text(pdf_docs)
                # Store raw text in session state for use in answering questions
                st.session_state.raw_text = raw_text
                # Process documents and update vector store
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, "text_faiss_index")
                # Optionally process code text as well
                if code_text.strip():
                    code_chunks = get_text_chunks(code_text)
                    get_vector_store(code_chunks, "code_faiss_index")
                # Generate and store quiz questions
                topics = get_topics_from_text(raw_text)
                st.session_state.quiz_questions = generate_questions(raw_text, topics)
                st.success("Content Processed")

    activity = st.selectbox("Choose an activity:", ["Ask a Question", "Take a Quiz","Answer Questions"])

    if activity == "Ask a Question":
        if 'question_count' not in st.session_state:
            st.session_state['question_count'] = 1

        for i in range(st.session_state['question_count']):
            user_question = st.text_input(f"Ask a Question {i+1}", key=f"question_{i}")
            if user_question:
                # Ensure user_input uses the updated vector store
                user_input(user_question)

        if st.button("Ask Another Question"):
            st.session_state['question_count'] += 1

    elif activity == "Take a Quiz":
        if 'quiz_questions' in st.session_state:
            display_quiz(st.session_state.quiz_questions)
        else:
            st.write("No quiz available. Please upload and process content first.")
    
    elif activity == "Answer Questions":
        if 'qa_pairs' not in st.session_state:
            st.session_state.qa_pairs = generate_question_answer_pairs(st.session_state.raw_text)

        for i, (q, a) in enumerate(st.session_state.qa_pairs):
            st.write(f"Question {i+1}: {q}")
            user_answer = st.text_area("Your Answer:", key=f"answer_{i}", height=150)
            if st.button("Evaluate Answer", key=f"evaluate_{i}"):
                score = evaluate_answer(user_answer, a)
                st.write(f"Score: {score} out of 5")


if __name__ == "__main__":
    main()













