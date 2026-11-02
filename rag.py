import re
import os
import chromadb

import time
from dotenv import load_dotenv

from openai import (
    OpenAI,
    APIError,
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
    BadRequestError,
    AuthenticationError,
    NotFoundError,
    PermissionDeniedError,
)
from sentence_transformers import SentenceTransformer
from pathlib import Path

class LLMError(Exception):
    pass

class EmptyLLMResponseError(Exception):
    pass

def format_error_message(error):
    return str(error).replace("\n", " ")[:500]

load_dotenv()

MAX_ATTEMPTS = 3

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing. Check your .env file.")

if not LLM_MODEL:
    raise ValueError("LLM_MODEL is missing. Check your .env file.")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    timeout=60.0,
    max_retries=0,
)




BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "ai_act_article5.txt"

VECTOR_STORE_DIR = BASE_DIR / "vector_store"

COLLECTION_NAME = "ai_act_article5_v2"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")



def chunk_article(text):

    parts = re.split(r'\n\n(\([a-z]\))\n\n', text) 
    chunks = [parts[0]]
    labels = parts[1::2]
    contents = parts[2::2]
    expected = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    pointer = 0 

    for label, content in zip(labels, contents):

        if pointer < len(expected) and label[1:-1] == expected[pointer]:

            chunks.append(' '.join([label , content]))
            pointer += 1
        
        else:
            chunks[-1] += '\n' + label + ' ' + content

    return chunks




def get_collection():

    client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    cosine_collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
    )

    return cosine_collection




def build_index():

    collection = get_collection()

    if collection.count() == 0:
        with open(DATA_FILE, "r") as file:
            text = file.read()

        chunks = chunk_article(text)
        embeddings = model.encode(chunks)
        collection.add(
        ids=[f'chunk_{i}' for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings
        )
   


def retrieve(query , n = 3):

    collection = get_collection()

    query_embedding = model.encode(query)
    results = collection.query(
            query_embeddings= [query_embedding],
            n_results=n
         )
    documents = results['documents'][0]
    distances = results['distances'][0]
    ids = results['ids'][0]

    return list(zip(ids, documents, distances))






def ask_llm(prompt):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            if not response.choices:
                response_details = format_error_message(
                    response.model_dump(exclude_none=True)
                )

                raise EmptyLLMResponseError(
                    f"The model returned no choices. "
                    f"Response details: {response_details}"
                )

            result = response.choices[0].message.content

            if not result or not result.strip():
                raise EmptyLLMResponseError(
                    "The model returned an empty response."
                )

            return result

        except (
            BadRequestError,
            AuthenticationError,
            NotFoundError,
        ) as e:
            raise LLMError(
                f"LLM request failed: {type(e).__name__}, "
                f"HTTP status: {getattr(e, 'status_code', None)}, "
                f"Details: {format_error_message(e)}"
            ) from e

        except PermissionDeniedError as e:
            if attempt >= 2:
                raise LLMError(
                    f"Permission denied after {attempt} attempts: "
                    f"{format_error_message(e)}"
                ) from e

            time.sleep(1)

        except (
            APITimeoutError,
            APIConnectionError,
            RateLimitError,
            InternalServerError,
        ) as e:
            if attempt == MAX_ATTEMPTS:
                raise LLMError(
                    f"LLM request failed after {attempt} attempts: "
                    f"{type(e).__name__}: "
                    f"{format_error_message(e)}"
                ) from e

            wait_time = 2 ** (attempt - 1)
            time.sleep(wait_time)

        except EmptyLLMResponseError as e:
            if attempt == MAX_ATTEMPTS:
                raise LLMError(
                    f"Invalid LLM response after {attempt} attempts: {format_error_message(e)}"
                ) from e

            time.sleep(1)

        except APIError as e:
            raise LLMError(
                f"LLM API request failed: {type(e).__name__}, "
                f"HTTP status: {getattr(e, 'status_code', None)}, "
                f"Details: {format_error_message(e)}"
            ) from e

    raise LLMError("LLM request failed after all attempts.")






def generate_answer(query, n=3):
    results = retrieve(query, n)

    documents = [document for chunk_id, document, distance in results]

    context = "\n\n".join(documents)

    prompt = f"""
Answer the question using only the context below.
If the answer is not in the context, say I don't know.

Context:
{context}

Question:
{query}
"""

    response = ask_llm(prompt)

    return response, context




def verify_answer(answer, context):

    prompt = f"""You are checking whether an answer is fully supported by a given context.
            If the answer correctly states that the context does not contain enough information to answer the question,
            treat this as GROUNDED, since accurately reporting an absence of information is itself a supported claim.

Context:
{context}

Answer:
{answer}

Is every claim in the answer supported by the context?
Respond with exactly one word on the first line — GROUNDED or NOT_GROUNDED — 
followed by a brief explanation on the next line."""

    result = ask_llm(prompt)

    lines = result.split('\n')
    clean_lines = []

    for line in lines:
        if line.strip():
            clean_lines.append(line.strip())

    first_line = clean_lines[0]

    if first_line.upper().startswith("NOT_GROUNDED"):
        verdict = "NOT_GROUNDED"
        remainder = first_line[len("NOT_GROUNDED"):]

    elif first_line.upper().startswith("GROUNDED"):
        verdict = "GROUNDED"
        remainder = first_line[len("GROUNDED"):]

    else:
        verdict = first_line.upper()
        remainder = ""

    explanation = (remainder.strip(" .") + " " + "\n".join(clean_lines[1:])).strip()

    return verdict, explanation
