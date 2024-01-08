from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import logging
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
import string
# import nltk

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# CORS middleware configuration
origins = [
    "http://localhost",
    "http://localhost:3000", 
    # Add more origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # Load SpaCy model for NER
# nlp_spacy = spacy.load("en_core_web_sm")

# Load Transformers pipeline for summarization
# Other models could be used here for better performance, but due to limited computation power I will use T5 here. You may replace it with another model.
summarizer = pipeline("summarization", model="t5-small")

# Load data and process it to extract data better suited for summarization
df = pd.read_csv("text_segments.csv")

#nltk.download('stopwords')

#stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Tokenization
    # tokens = word_tokenize(text)
    
    # # Remove stopwords
    # tokens = [token for token in tokens if token not in stop_words]
    
    return text

df = df.drop_duplicates(subset=['text'])

df['processed_text'] = df['text'].apply(preprocess_text)

#This is just here for testing purposes
@app.get("/")
def read_root():
    return {"Hello": "World"}

class Document(BaseModel):
    content: str

# # Function to split text into chunks
# def split_text_into_chunks(text, chunk_size=512, max=None):
#     chunks = [text[i:i+chunk_size] for i in range(0, len(text) if max is None else max, chunk_size)]
#     return chunks

# # Function to summarize each chunk and concatenate the results
# def summarize_text(text, chunk_size=512, max_length=50, min_length=20, length_penalty=2.0, num_beams=4):
#     chunks = split_text_into_chunks(text, chunk_size, 4096)
#     summaries = [summarizer(chunk, max_length=max_length, min_length=min_length, length_penalty=length_penalty, num_beams=num_beams)[0]['summary_text'] for chunk in chunks]
#     return ' '.join(summaries)

#This model supports only max_length=512. If you switch models, you can increase max length and hence increase functionality. I have limited Computation Power.
def summarize_text(text, max_length=512, max_summary_length=50, min_length=20, length_penalty=2.0, num_beams=4):
    # Truncate the text if max_length is specified
    truncated_text = text[:max_length] if max_length else text
    # Summarize the truncated text
    summary = summarizer(truncated_text, max_length=max_summary_length, min_length=min_length, length_penalty=length_penalty, num_beams=num_beams)[0]['summary_text']
    return summary

@app.get("/summarize/{doc_name}")
def summarize(doc_name: str):
    try:
        logging.info("This should be logged")
        # Extract text from the specified document
        doc_groups = df.groupby('doc_name')['text'].apply(lambda x: ' '.join(map(str, x))).reset_index()
        logging.info("extracted text")
        # Load a pre-trained sentence embedding model
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        logging.info("read model")

        # Generate sentence embeddings for each document
        doc_groups['embedding'] = doc_groups['text'].apply(lambda x: model.encode(x))
        logging.info("generated embeddings")

        logging.info("loaded summary model")

        # Generate summaries for each document
        doc_groups['summary'] = doc_groups['text'].apply(lambda x: summarize_text(x))
        logging.info("generated summaries")

        summary = doc_groups[doc_groups['doc_name'] == doc_name]['summary'].values[0]
        logging.info("read doc name's summary")

        return {"summary": summary}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error summarizing document: {str(e)}"
        )

# # here we could use SpaCy's named entity recognition, This would be where topic-identification could be added in the future.
# @app.get("/topic-identification/{doc_name}")
# def identify_topics(doc_name: str):
#     try:
#         # implementation logic to be added here

#     except Exception as e:
#         raise HTTPException(
#             status_code=500, detail=f"Error identifying topics: {str(e)}"
#         )
    
    
############## Example for implementation of question answering ##################
    #not fully implemented

# # Load Hugging Face model for question answering
# qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
       
# @app.get("/question-answering/{doc_name}")
# def question_answering(doc_name: str, question: str):
#     try:
#         # Extract text from the specified document
#         result = qa_pipeline(question=question, context=doc.content)
#         return {"answer": result["answer"], "confidence": result["score"]}


#     except Exception as e:
#         raise HTTPException(
#             status_code=500, detail=f"Error processing question: {str(e)}"
#         )