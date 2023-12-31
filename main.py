from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import logging

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

# Load SpaCy model for NER
nlp_spacy = spacy.load("en_core_web_sm")

# Load Hugging Face model for question answering
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Load Transformers pipeline for summarization
summarizer = pipeline("summarization", model="t5-small")

df = pd.read_csv("text_segments.csv")

@app.get("/")
def read_root():
    return {"Hello": "World"}

class Document(BaseModel):
    content: str

# @app.post("/ner")
# def extract_named_entities(doc: Document):
#     doc_spacy = nlp_spacy(doc.content)
#     entities = [{"text": ent.text, "label": ent.label_} for ent in doc_spacy.ents]
#     return {"entities": entities}

# @app.post("/question-answering")
# def answer_question(doc: Document, question: str = Query(..., title="Question")):
#     result = qa_pipeline(question=question, context=doc.content)
#     return {"answer": result["answer"], "confidence": result["score"]}


@app.get("/question-answering/{doc_name}")
def question_answering(doc_name: str, question: str):
    try:
        # Extract text from the specified document
        text = df.loc[df['doc_name'] == doc_name, 'text'].iloc[0]

        # Use SpaCy for question answering
        doc = nlp(text)
        answer = " ".join(token.text for token in doc if token.is_alpha)

        return {"answer": answer}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing question: {str(e)}"
        )

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

@app.get("/topic-identification/{doc_name}")
def identify_topics(doc_name: str):
    try:
        # Extract text from the specified document
        text = df.loc[df['doc_name'] == doc_name, 'text'].iloc[0]

        # Implement your topic identification logic here
        # For example, you could use SpaCy's named entity recognition (NER)
        doc = nlp(text)
        topics = [ent.text for ent in doc.ents]

        return {"topics": topics}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error identifying topics: {str(e)}"
        )