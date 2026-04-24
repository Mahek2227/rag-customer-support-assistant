from langgraph.graph import StateGraph
from langchain_community.llms import Ollama


class GraphState(dict):
    pass


def process_query(state, vectorstore):
    query = state.get("query", "")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    docs = retriever.invoke(query)

    # DEBUG (optional but useful)
    print("\n[DEBUG] Retrieved docs:", len(docs))

    context = "\n\n".join([doc.page_content for doc in docs])

    llm = Ollama(model="llama3")

    prompt = f"""
You are a helpful AI assistant.

Use the context below to answer the question.
If context is weak, still give a reasonable answer.

-------------------
Context:
{context}
-------------------

Question: {query}

Answer:
"""

    answer = llm.invoke(prompt)

    confidence = len(docs) / 5

    return {
        **state,
        "query": query,
        "answer": answer,
        "confidence": confidence
    }


def route(state):
    if state.get("confidence", 0) < 0.3:
        return "human"
    return "output"


def human_node(state):
    print("\n⚠️ Escalated to Human Support")
    human_answer = input("Enter human response: ")

    return {
        **state,
        "answer": human_answer,
        "confidence": 1.0
    }


def output_node(state):
    print("\n✅ Final Answer:\n")
    print(state.get("answer", "No answer found"))
    return state


def build_graph(vectorstore):
    workflow = StateGraph(GraphState)

    def process_wrapper(state):
        return process_query(state, vectorstore)

    workflow.add_node("process", process_wrapper)
    workflow.add_node("human", human_node)
    workflow.add_node("output", output_node)

    workflow.set_entry_point("process")

    workflow.add_conditional_edges(
        "process",
        route,
        {
            "human": "human",
            "output": "output"
        }
    )

    workflow.add_edge("human", "output")

    return workflow.compile()