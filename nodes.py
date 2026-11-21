from config import llm
from state import ResearchState
from tools import search_tool
import json
from langchain.messages import AIMessage




def save_response_node(state:ResearchState):
    answer = state["answer"]
    return {
        "messages": [
            AIMessage(content=answer)
        ]
    }

# Helper function
def format_conversation(messages):
    conversation = []
    for message in messages:
        role = message.type.upper()
        content = message.content
        conversation.append(
            f"{role}: {content}"
        )
    return "\n".join(conversation)



# Task Specialization

def analyzer_node(state: ResearchState):

    question = state["question"]
    messages = state.get("messages", [])

    history = format_conversation(messages)

    prompt = f"""
    You are an intelligent research assistant.

    Previous conversation:
    {history}

    Latest user question:
    {question}

    Understand the latest question using the previous
    conversation as context.

    If the user asks a follow-up question, identify what
    they are referring to from the previous conversation.

    Identify the user's actual research objective.
    Explain what information needs to be researched.

    If the question is independent of previous messages,
    analyze it independently.

    Return a clear analysis.
    """

    response = llm.invoke(prompt)

    return {
        "analysis": response.content
    }

    
    



def planer_node(state:ResearchState):
    
    question = state['question']
    analysis = state['analysis']
    
    prompt = f"""
    You are an expert research planner.

    Original Question:
    {question}

    Query Analysis:
    {analysis}

    Break this research task into 3 to 5
    specific and independent search queries.

    Requirements:
    1. Cover all important aspects.
    2. Avoid duplicate queries.
    3. Each query should be useful for research.
    4. Keep queries specific.

    Return one query per line.
    Do not include numbering or explanations.
    """
    response = llm.invoke(prompt)
    queries = []
    for line in response.content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("-"):
            line = line[1:].strip()
        queries.append(line)
    
    return {
        "research_plan" : queries
    }
    
 
 
 # Search the web   
def researcher_node(state:ResearchState):
    
    queries = state['research_plan']
    all_results = []
    for query in queries:
        results = search_tool(query)
        
        for result in results:
            all_results.append({
                "title": result["title"],
                "content": result["content"],
                "url": result["url"]
            })
    
    return {
        'research_results' : all_results
    }
    


# Generate answer based on the search results and question

def generator_node(state: ResearchState):

    question = state["question"]
    analysis = state["analysis"]
    research = state["research_results"]
    feedback = state.get("feedback", "")

    messages = state.get("messages", [])
    history = format_conversation(messages)

    prompt = f"""
    You are an expert research assistant.

    Previous conversation:
    {history}

    Latest user question:
    {question}

    Question analysis:
    {analysis}

    Research evidence:
    {research}

    Evaluator feedback:
    {feedback}

    Instructions:

    1. Answer the latest question.
    2. Use previous conversation when relevant.
    3. Use the provided research evidence.
    4. Do not invent unsupported facts.
    5. If the user asks a follow-up question,
       maintain continuity with the previous discussion.
    6. Provide a clear and structured answer.

    Generate the final answer.
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }

    
    
    

# Answer evaluation

def evaluator_node(state: ResearchState):

    question = state["question"]
    answer = state["answer"]
    research_results = state["research_results"]

    prompt = f"""
    You are a strict AI answer evaluator.

    Original User Question:
    {question}

    Research Evidence:
    {research_results}

    Generated Answer:
    {answer}

    Evaluate the answer using these criteria:

    1. Accuracy:
       Are the claims supported by the research?
       Identify potentially incorrect claims.

    2. Completeness:
       Does the answer cover all important parts
       of the user's question?

    3. Relevance:
       Does the answer directly address the question?

    4. Clarity:
       Is the explanation understandable and organized?

    5. Evidence:
       Are factual claims supported by the research?

    Approval rules:
    - Approve only if the answer is sufficiently
      complete, relevant and supported.
    - Reject if important information is missing.
    - Reject if important claims are unsupported.
    - Give specific feedback explaining what to improve.
    - Do not reject merely because of writing style.

    Return ONLY valid JSON in this exact format:

    {{
        "approved": true,
        "feedback": "Your evaluation feedback"
    }}
    """

    response = llm.invoke(prompt)

    try:
        evaluation = json.loads(response.content)

        approved = evaluation["approved"]
        feedback = evaluation["feedback"]

        if not isinstance(approved, bool):
            raise ValueError("approved must be boolean")

        if not isinstance(feedback, str):
            raise ValueError("feedback must be a string")

    except (ValueError, KeyError, TypeError):
        approved = False
        feedback = (
            "Evaluator returned an invalid response. "
            "Please generate and evaluate again."
        )

    # This return must be outside the try/except
    return {
        "approved": approved,
        "feedback": feedback,
        "iteration": state.get("iteration", 0) + 1
    }
