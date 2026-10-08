import logging
from typing import Any, Literal, cast

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph

from app.core.config import get_settings
from app.rag.state import AgentState, EvidenceGrade, RouteDecision
from app.rag.vectorstore import get_retriever


logger = logging.getLogger(__name__)

settings = get_settings()

_llm: Any = None
_web_search: Any = None


def llm() -> Any:
    """Create and cache the Groq chat model."""
    global _llm

    if _llm is None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing")

        _llm = ChatGroq(
            model=settings.groq_model,
            temperature=0,
            api_key=settings.groq_api_key,
        )

    return _llm


def web_search_tool() -> Any:
    """Create and cache the Tavily search tool."""
    global _web_search

    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")

        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )

    return _web_search


def add_trace(state: AgentState, message: str) -> list[str]:
    """Append a message to the execution trace."""
    return [*state.get("trace", []), message]


def route_question(state: AgentState) -> dict[str, Any]:
    """Route the question to the private KB or direct response."""

    router = llm().with_structured_output(
        RouteDecision,
        method="json_mode",
    )

    decision = cast(
        RouteDecision,
        router.invoke(
            f"""
You route messages for an enterprise IT support assistant.

Use "kb" for questions about:

- Company IT policies
- VPN
- Password reset
- MFA
- Laptop setup
- Software access
- Security
- Email
- Devices
- Troubleshooting
- Technology support

Use "direct" only for:

- Greetings
- Thanks
- Casual conversation
- Questions that require no company knowledge

Question:

{state["question"]}

Return valid JSON:

{{"route": "kb"}}

or

{{"route": "direct"}}
"""
        ),
    )

    return {
        "source_used": decision.route,
        "trace": add_trace(
            state,
            f"Router → {decision.route.upper()}",
        ),
    }


def route_after_router(state: AgentState,) -> Literal["retrieve_kb", "direct_answer"]:
    """Determine the next node after routing."""

    if state["source_used"] == "kb":
        return "retrieve_kb"

    return "direct_answer"


def retrieve_kb(state: AgentState) -> dict[str, Any]:
    """Retrieve relevant documents from the private knowledge base."""

    docs = get_retriever().invoke(
        state["current_query"]
    )

    return {
        "kb_docs": docs,
        "trace": add_trace(
            state,
            f"Private KB retrieval → {len(docs)} chunks",
        ),
    }


def grade_kb(state: AgentState) -> dict[str, Any]:
    """Evaluate whether the private KB contains sufficient evidence."""

    grader = llm().with_structured_output(
        EvidenceGrade,
        method="json_mode",
    )

    context = "\n\n".join(
        (
            f"Source: {document.metadata.get('source', 'unknown')}\n"
            f"{document.page_content}"
        )
        for document in state["kb_docs"]
    )

    grade = cast(
        EvidenceGrade,
        grader.invoke(
            f"""
You grade evidence for an enterprise IT support assistant.

Question:
{state["question"]}

Private company KB evidence:
{context}

Return "good" only if the evidence is sufficient to answer
the question confidently and specifically.

Return "weak" if the evidence is missing, incomplete,
ambiguous, or unrelated.

Return valid JSON:

{{"grade": "good"}}

or

{{"grade": "weak"}}
"""
        ),
    )

    return {
        "kb_grade": grade.grade,
        "trace": add_trace(
            state,
            f"KB evidence grade → {grade.grade.upper()}",
        ),
    }


def after_kb(state: AgentState,) -> Literal["generate_from_kb", "search_web"]:
    """Determine whether to answer from KB or use web fallback."""

    if state["kb_grade"] == "good":
        return "generate_from_kb"

    return "search_web"


def search_web(state: AgentState) -> dict[str, Any]:
    """Search external web sources using Tavily."""

    result: Any = web_search_tool().invoke(
        {
            "query": state["current_query"],
        }
    )

    lines: list[str] = []
    citations: list[dict[str, str]] = []

    if isinstance(result, dict):
        result_data = cast(dict[str, Any], result)

        answer = result_data.get("answer")
        if answer:
            lines.append(
                "Search answer: "
                + str(answer)
            )

        raw_results = result_data.get("results", [])

        if isinstance(raw_results, list):
            for raw_item in raw_results:
                if not isinstance(raw_item, dict):
                    continue

                item = cast(dict[str, Any], raw_item)

                title = str(item.get("title", ""))
                url = str(item.get("url", ""))
                content = str(item.get("content", ""))

                lines.append(
                    f"Title: {title}\n"
                    f"URL: {url}\n"
                    f"Content: {content}"
                )

                citations.append(
                    {
                        "title": title or url,
                        "url": url,
                        "type": "web",
                    }
                )

    else:
        lines.append(str(result))

    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(
            state,
            "Web fallback → Tavily search",
        ),
    }


def grade_web(state: AgentState) -> dict[str, Any]:
    """Evaluate whether web evidence is sufficient."""

    grader = llm().with_structured_output(
        EvidenceGrade,
        method="json_mode",
    )

    grade = cast(
        EvidenceGrade,
        grader.invoke(
            f"""
You evaluate web evidence for an enterprise IT support assistant.

Question:
{state["question"]}

Web evidence:
{state["web_results"]}

Return "good" if the evidence is sufficient and directly
relevant to the question.

Return "weak" if the evidence is insufficient,
ambiguous, or unrelated.

Return valid JSON:

{{"grade": "good"}}

or

{{"grade": "weak"}}
"""
        ),
    )

    return {
        "web_grade": grade.grade,
        "trace": add_trace(
            state,
            f"Web evidence grade → {grade.grade.upper()}",
        ),
    }


def after_web(state: AgentState,) -> Literal["generate_from_web","rewrite_query","insufficient",]:
    """Determine whether to answer, retry, or stop."""

    if state["web_grade"] == "good":
        return "generate_from_web"

    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"

    return "insufficient"


def rewrite_query(state: AgentState) -> dict[str, Any]:
    """Rewrite the query for improved retrieval."""

    response = llm().invoke(
        f"""
Rewrite this IT support question for better private
knowledge retrieval and vendor web search.

Requirements:

- Preserve the original intent.
- Add useful technical keywords.
- Make the query specific.
- Do not answer the question.
- Return only the rewritten query.

Question:{state["question"]}
"""
    )

    rewritten = str(response.content).strip()

    return {
        "current_query": rewritten,
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(state,f"Query rewrite → {rewritten}",),
    }


def generate_from_kb(state: AgentState) -> dict[str, Any]:
    """Generate an answer using private KB evidence only."""

    context = "\n\n".join(
        (
            f"[Source: "
            f"{document.metadata.get('source', 'unknown')}]\n"
            f"{document.page_content}"
        )
        for document in state["kb_docs"]
    )

    response = llm().invoke(
        f"""
You are an enterprise IT support copilot.
Answer ONLY from the private company KB below.

Rules:
11. Be concise and actionable.
2. If steps are provided, present them clearly.
3. Do not invent policy details.
4. Do not use external knowledge.
5. If the KB does not contain an answer, do not guess.
6. Mention that the answer is based on the company's
   private knowledge base.
7. Return the answer as plain text only.
8. Do not use Markdown formatting.
9. Do not use *, **, _, #, backticks, or other Markdown symbols.
10. Use simple numbered lists when necessary.

Question:
{state["question"]}

Private company KB:
{context}
"""
    )

    answer = str(response.content)

    citations: list[dict[str, str]] = []
    seen: set[str] = set()

    for document in state["kb_docs"]:
        source_value = document.metadata.get(
            "source",
            "Private KB",
        )
        source = str(source_value)

        if source not in seen:
            seen.add(source)

            citations.append(
                {
                    "title": source.split("/")[-1],
                    "url": "",
                    "type": "private_kb",
                }
            )

    return {
        "answer": answer,
        "source_used": "private_kb",
        "citations": citations,
        "trace": add_trace(
            state,
            "Answer generation → PRIVATE KB",
        ),
    }


def generate_from_web(state: AgentState) -> dict[str, Any]:
    """Generate an answer using web evidence only."""

    response = llm().invoke(
        f"""
You are an enterprise IT support copilot.

The private company knowledge base was insufficient.

Answer ONLY from the web evidence below.

Rules:

1. Be concise and actionable.
2. If steps are provided, present them clearly.
3. Do not invent policy details.
4. Do not use external knowledge.
5. If the KB does not contain an answer, do not guess.
6. Mention that the answer is based on the company's
   private knowledge base.
7. Return the answer as plain text only.
8. Do not use Markdown formatting.
9. Do not use *, **, _, #, backticks, or other Markdown symbols.
10. Use simple numbered lists when necessary.

Question:
{state["question"]}

External web evidence:
{state["web_results"]}
"""
    )

    return {
        "answer": str(response.content),
        "source_used": "web_search",
        "trace": add_trace(
            state,
            "Answer generation → WEB SEARCH",
        ),
    }


def direct_answer(state: AgentState) -> dict[str, Any]:
    """Handle greetings and casual conversation."""

    response = llm().invoke(
        f"""
Return the answer as plain text only.
Do not use Markdown formatting.
Do not use *, **, _, #, backticks, or other Markdown symbols.
Use simple numbered lists when necessary.

Respond briefly and naturally to the following message.

Message:
{state["question"]}
"""
    )

    return {
        "answer": str(response.content),
        "source_used": "direct",
        "trace": add_trace(
            state,
            "Direct response → no retrieval",
        ),
    }


def insufficient(state: AgentState) -> dict[str, Any]:
    """Return a safe response when evidence is insufficient."""

    return {
        "answer": (
            "I couldn't find enough reliable evidence in the "
            "company knowledge base or external search to "
            "answer confidently. Please contact the IT help "
            "desk or provide more details."
        ),
        "source_used": "insufficient_evidence",
        "trace": add_trace(
            state,
            "Stopped → insufficient reliable evidence",
        ),
    }


def build_graph() -> Any:
    """Build and compile the LangGraph workflow."""

    graph = StateGraph(AgentState)

    nodes = {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }

    for name, function in nodes.items():
        graph.add_node(
            name,
            function,
        )

    # Entry point
    graph.add_edge(
        START,
        "route_question",
    )

    # Route question
    graph.add_conditional_edges(
        "route_question",
        route_after_router,
        {
            "retrieve_kb": "retrieve_kb",
            "direct_answer": "direct_answer",
        },
    )

    # Private KB retrieval
    graph.add_edge(
        "retrieve_kb",
        "grade_kb",
    )

    # Grade private KB
    graph.add_conditional_edges(
        "grade_kb",
        after_kb,
        {
            "generate_from_kb": "generate_from_kb",
            "search_web": "search_web",
        },
    )

    # Web search
    graph.add_edge(
        "search_web",
        "grade_web",
    )

    # Grade web evidence
    graph.add_conditional_edges(
        "grade_web",
        after_web,
        {
            "generate_from_web": "generate_from_web",
            "rewrite_query": "rewrite_query",
            "insufficient": "insufficient",
        },
    )

    # Retry loop
    graph.add_edge(
        "rewrite_query",
        "retrieve_kb",
    )

    # Terminal nodes
    graph.add_edge(
        "generate_from_kb",
        END,
    )

    graph.add_edge(
        "generate_from_web",
        END,
    )

    graph.add_edge(
        "direct_answer",
        END,
    )

    graph.add_edge(
        "insufficient",
        END,
    )

    return graph.compile()


agent_graph = build_graph()


def ask(question: str) -> Any:
    """Run the enterprise IT support workflow."""

    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }

    return agent_graph.invoke(initial)


async def ask_stream(question: str):
    """Run the enterprise IT support workflow with streaming."""
    
    initial = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }
    
    # Generation node names that produce the final answer
    generation_nodes = {"direct_answer", "generate_from_kb", "generate_from_web"}
    
    # Track if we've started streaming the answer
    answer_started = False
    
    async for event in agent_graph.astream_events(initial, version="v2"):
        event_type = event["event"]
        name = event.get("name", "")
        data = event.get("data", {})
        metadata = event.get("metadata", {})
        
        # Stream tokens from generation nodes
        if event_type == "on_chat_model_stream" and "chunk" in data:
            chunk = data["chunk"]
            langgraph_node = metadata.get("langgraph_node", "")
            
            if langgraph_node in generation_nodes and hasattr(chunk, "content") and chunk.content:
                if not answer_started:
                    yield {"type": "start"}
                    answer_started = True
                
                yield {"type": "token", "content": chunk.content}
        
        # When the graph completes, send metadata
        elif event_type == "on_chain_end" and name == "LangGraph":
            output = data.get("output", {})
            
            # Send metadata after streaming completes
            yield {
                "type": "metadata",
                "source_used": output.get("source_used", ""),
                "citations": output.get("citations", []),
                "trace": output.get("trace", []),
                "rewritten_query": output.get("current_query", question),
            }
            
            yield {"type": "done"}
            break