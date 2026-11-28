from graph import app
from langchain_core.messages import HumanMessage


def chat(question, thread_id):

    # Identify the conversation
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # Prepare the state for this turn
    initial_state = {
        "question": question,
        "analysis": "",
        "research_plan": [],
        "research_results": [],
        "answer": "",
        "feedback": "",
        "approved": False,
        "iteration": 0,

        # Add the latest user message
        "messages": [
            HumanMessage(content=question)
        ]
    }

    # Execute the graph
    result = app.invoke(
        initial_state,
        config=config
    )

    return result["answer"]


if __name__ == "__main__":

    thread_id = "conversation_001"

    print("\nAI Research Assistant")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        answer = chat(question, thread_id)

        print("\nAI:", answer)
        print("\n" + "=" * 60 + "\n")
