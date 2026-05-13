PROMPT_TEMPLATE = """

QUESTION:
{question}

CONTEXT:
{context}
""".strip()


class RAGBase:

    def __init__(
        self,
        es_client,
        index_name,
        llm_client,
        instructions,
        course='llm-zoomcamp',
        prompt_template=PROMPT_TEMPLATE,
        model='gpt-5.4-mini',
        top_k=5
    ):
        self.es = es_client
        self.index_name = index_name

        self.llm_client = llm_client
        self.instructions = instructions

        self.course = course
        self.prompt_template = prompt_template
        self.model = model
        self.top_k = top_k

    def search(self, query, top_k=None):
        top_k = top_k or self.top_k

        res = self.es.search(
            index=self.index_name,
            size=top_k,
            query={
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": ["question^3", "section^2", "answer"]
                        }
                    },
                    "filter": [
                        {"term": {"course": self.course}}
                    ]
                }
            }
        )

        return [hit["_source"] for hit in res["hits"]["hits"]]

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc['section'])
            lines.append('Q: ' + doc['question'])
            lines.append('A: ' + doc['answer'])
            lines.append('')

        return '\n'.join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        return response.output_text
    
    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer