# **EduBot - AI-Powered Educational Assistant** 🤖📚

## **Overview**  
**EduBot** is an AI-powered **educational assistant** designed to process **PDF documents**, generate **quiz questions**, and answer user queries. It leverages **Natural Language Processing (NLP)** and **machine learning models** to extract key insights from educational content and provide **interactive learning experiences**.

## **Key Features**  

🔹 **PDF Text Extraction**: Supports **multiple PDFs**, extracts **text and code blocks** using `pdfminer` and `PyMuPDF`.  
🔹 **AI-Powered Q&A**: Uses **Google Generative AI (Gemini-Pro)** to **answer user queries** based on the extracted content.  
🔹 **Quiz Generation**: Automatically generates **multiple-choice questions (MCQs)** from processed text.  
🔹 **Key Topic Identification**: Extracts **important keywords and phrases** using `KeyBERT` and **POS tagging**.  
🔹 **Wikipedia Integration**: Fetches **additional learning resources** for weak topics.  
🔹 **Vector Search**: Utilizes `FAISS` for **efficient similarity-based search** within the extracted text.  
🔹 **Streamlit UI**: Provides an **interactive web interface** for users to **upload PDFs, ask questions, and take quizzes**.

---

## **Project Workflow**  

1. **Upload PDF Documents**:  
   - Users upload educational PDFs.  
   - Extracts text using **pdfminer** and **PyMuPDF** (fallback).  
   - Identifies **code snippets separately** for programming-related content.

2. **Process Content**:  
   - Tokenizes text using **NLTK**.  
   - Identifies key topics using **KeyBERT & POS tagging**.  
   - Stores extracted text in **FAISS vector store** for **efficient searching**.

3. **AI-Powered Q&A System**:  
   - Uses **Google Gemini-Pro AI** to provide **detailed, context-aware answers**.  
   - Generates relevant **question-answer pairs** based on extracted content.

4. **Quiz Generation & Evaluation**:  
   - Automatically generates **fill-in-the-blank and MCQ quizzes**.  
   - Evaluates **user responses** and provides **feedback**.  
   - Fetches additional resources from **Wikipedia** for incorrect answers.

---

## **Technologies Used**  

🟢 **Programming Language**: Python  
🟢 **Framework**: Streamlit (for UI)  
🟢 **NLP & AI Models**:  
   - `Google Generative AI (Gemini-Pro)` – AI-powered Q&A  
   - `KeyBERT` – Extracts key phrases  
   - `NLTK` – Text tokenization & processing  
🟢 **PDF Processing**:  
   - `pdfminer.six` – Extracts text from PDFs  
   - `PyMuPDF` – Fallback PDF text extraction  
🟢 **Vector Search**: `FAISS` – Fast retrieval of relevant text  
🟢 **Wikipedia API** – Fetches additional learning resources  
🟢 **Environment Management**: `dotenv` (Loads API keys)

---

## **Installation & Setup**  

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/EduBot.git
cd EduBot
```

### **2. Create a Virtual Environment (Optional)**
```bash
python -m venv edubot_env
source edubot_env/bin/activate  # Mac/Linux
edubot_env\Scripts\activate  # Windows
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Set Up API Key for Google AI**
1. Create an `.env` file in the root directory.
2. Add the **Google Generative AI API Key**:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

### **5. Run the Application**
```bash
streamlit run EduBot.py
```

---

## **How to Use**  

1️⃣ **Upload PDFs** in the sidebar.  
2️⃣ Click **Submit & Process** to extract **text and key topics**.  
3️⃣ Choose **Ask a Question** to enter **educational queries**.  
4️⃣ Select **Take a Quiz** to answer **auto-generated MCQs**.  
5️⃣ Review **incorrect answers** and explore **Wikipedia resources**.

---

## **Future Enhancements 🚀**  

✅ **Speech-to-Text Integration** for **voice-based queries**.  
✅ **AI-Powered Summarization** for quick learning.  
✅ **Interactive Chatbot** for real-time student engagement.  
✅ **Real-time Document Parsing** from **Google Drive & Cloud Storage**.

---

## **License**  

📜 This project is licensed under the **MIT License**.
