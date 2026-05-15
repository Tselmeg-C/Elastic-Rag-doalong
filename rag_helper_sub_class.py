import inspect
from typing import Any, Callable

PROMPT_TEMPLATE = """
QUESTION:
{question}

CONTEXT:
{context}
""".strip()


class RAGBase:
    """Backend-agnostic RAG orchestration.

    This class does not assume Elasticsearch or any specific retrieval backend.
    Inject a search function via `search_fn`.
    """

    def __init__(
        self,
        llm_client: Any,
        instructions: str,
        search_fn: Callable[..., list[dict]],
        course: str = "llm-zoomcamp",
        prompt_template: str = PROMPT_TEMPLATE,
        model: str = "gpt-5.4-mini",
        top_k: int = 5,
    ):
        self.llm_client = llm_client
        self.instructions = instructions
        self.search_fn = search_fn
        self.course = course
        self.prompt_template = prompt_template
        self.model = model
        self.top_k = top_k

    def _call_search_fn(self, query: str, course: str, top_k: int) -> list[dict]:
        """Call injected search function with compatible parameters.

        Supports:
        - search(query, course=..., top_k=...)
        - search(query, num_results=...)
        - search(query)
        """
        sig = inspect.signature(self.search_fn)
        params = sig.parameters

        kwargs: dict[str, Any] = {}
        if "course" in params:
            kwargs["course"] = course
        if "top_k" in params:
            kwargs["top_k"] = top_k
        elif "num_results" in params:
            kwargs["num_results"] = top_k

        return self.search_fn(query=query, **kwargs)

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
        resolved_course = course or self.course
        resolved_top_k = top_k or self.top_k
        search_results = self._call_search_fn(query=query, course=resolved_course, top_k=resolved_top_k)
        prompt = self.build_prompt(query, search_results)
        return self.llm(prompt)


class ElasticRAG(RAGBase):
    """Elasticsearch-backed RAG implementation.

    Requires ES client and index name, and wires `search_elastic` as default search.
    """

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

        super().__init__(
            llm_client=llm_client,
            instructions=instructions,
            search_fn=search_fn or self.search_elastic,
            course=course,
            prompt_template=prompt_template,
            model=model,
            top_k=top_k,
        )

    def search_elastic(self, query: str, course: str | None = None, top_k: int | None = None) -> list[dict]:
        resolved_course = course or self.course
        size = top_k or self.top_k

        search_query = {
            "size": size,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["question^3.0", "section^0.5", "answer^3.0"],
                                "type": "best_fields",
                            }
                        }
                    ],
                    "filter": [{"term": {"course": resolved_course}}],
                }
            },
        }

        res = self.es.search(index=self.index_name, body=search_query)
        return [hit["_source"] for hit in res["hits"]["hits"]]

class MinSearchRAG(RAGBase):
    def __init__(
        self,
        index: Any,  # minsearch index object
        llm_client: Any,
        instructions: str,
        course: str = "llm-zoomcamp",
        prompt_template: str = PROMPT_TEMPLATE,
        model: str = "gpt-5.4-mini",
        top_k: int = 5,
    ):
        self.index = index

        super().__init__(
            llm_client=llm_client,
            instructions=instructions,
            search_fn=self.search_minsearch,  # bind subclass search
            course=course,
            prompt_template=prompt_template,
            model=model,
            top_k=top_k,
        )

    def search_minsearch(
        self, query: str, course: str | None = None, top_k: int | None = None
    ) -> list[dict]:
        resolved_course = course or self.course
        resolved_top_k = top_k or self.top_k

        boost_dict = {"question": 3.0, "section": 0.5}
        filter_dict = {"course": resolved_course}

        return self.index.search(
            query,
            num_results=resolved_top_k,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
        )