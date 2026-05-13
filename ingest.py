import requests
from elasticsearch import Elasticsearch
from tqdm import tqdm


INDEX_NAME = "faq_documents"

def load_faq_data():
    docs_url = 'https://datatalks.club/faq/json/courses.json'
    response = requests.get(docs_url)
    courses_raw = response.json()

    documents = []
    url_prefix = 'https://datatalks.club/faq'

    for course in courses_raw:
        course_url = f'{url_prefix}{course["path"]}'
        course_response = requests.get(course_url)
        course_response.raise_for_status()
        course_data = course_response.json()

        documents.extend(course_data)

    return documents


def build_index(documents):
    es = Elasticsearch("http://localhost:9200")

    mapping = {
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "course": {"type": "keyword"},
                "section": {"type": "text"},
                "question": {"type": "text"},
                "answer": {"type": "text"}
            }
        }
    }

    # Delete existing index if exists
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)

    # Create fresh index
    es.indices.create(index=INDEX_NAME, body=mapping)

    # Index documents
    for doc in tqdm(documents):
        es.index(
            index=INDEX_NAME,
            id=doc.get("id"),
            document=doc
        )

    # Refresh so docs become searchable immediately
    es.indices.refresh(index=INDEX_NAME)

    return es