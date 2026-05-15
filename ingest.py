import requests
from elasticsearch import Elasticsearch
from tqdm import tqdm
from minsearch import Index


# INDEX_NAME = "faq_documents"

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


def build_elasticsearch_index(documents,index_name):
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
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    # Create fresh index
    es.indices.create(index=index_name, body=mapping)

    # Index documents
    for doc in tqdm(documents):
        es.index(
            index=index_name,
            id=doc.get("id"),
            document=doc
        )

    # Refresh so docs become searchable immediately
    es.indices.refresh(index=index_name)

    return es

def build_minsearch_index(documents):
    index = Index(
        text_fields=['question', 'section', 'answer'],
        keyword_fields=['course']
    )
    index.fit(documents)
    return index