from typing import Any, Callable

PROMPT_TEMPLATE = """
QUESTION:
{question}

CONTEXT:
{context}
""".strip()


class RAGBase:
    def __init__(
        self,
        es_client: Any,
        index_name: str,
        llm_client: Any,
        instructions: str,
        course: str = "llm-zoomcamp",
        prompt_template: str = PROMPT_TEMPLATE,
        model: str = "gpt-5.4-mini",
        top_k: int = 5,
        search_fn: Callable[..., list[dict]] | None = None,
    ):
        self.es = es_client
        self.index_name = index_name
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model
        self.top_k = top_k

        # Strategy injection: if None, use Elasticsearch search by default.
        self.search_fn = search_fn or self.search_elastic

    def search_elastic(self, query: str, course: str | None = None, top_k: int | None = None) -> list[dict]:
        course = course or self.course
        size = top_k or self.top_k

        search_query = {
            "size": size,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "question^3.0",
                                    "section^0.5",
                                    "answer^3.0",
                                ],
                                "type": "best_fields",
                            }
                        }
                    ],
                    "filter": [{"term": {"course": course}}],
                }
            },
        }

        res = self.es.search(index=self.index_name, body=search_query)
        return [hit["_source"] for hit in res["hits"]["hits"]]

    def build_context(self, search_results: list[dict]) -> str:
        lines = []
        for doc in search_results:
            lines.append(doc.get("section", ""))
            lines.append("Q: " + doc.get("question", ""))
            lines.append("A: " + doc.get("answer", ""))
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query: str, search_results: list[dict]) -> str:
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)

    def llm(self, prompt: str) -> str:
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages,
        )

        return response.output_text

    def rag(self, query: str, course: str | None = None, top_k: int | None = None) -> str:
        search_results = self.search_fn(query=query, course=course or self.course, top_k=top_k or self.top_k)
        prompt = self.build_prompt(query, search_results)
        return self.llm(prompt)
