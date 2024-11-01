from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.schema.messages import SystemMessage, HumanMessage
from langchain.tools import tool
from typing import List, Tuple, Dict
from dotenv import load_dotenv
import json

load_dotenv()

@tool
def solve_knapsack(problem_input: str) -> str:
    """
    Solves the knapsack problem.
    :param problem_input: A string containing items, weights, values, and capacity in the format:
        'items=['A', 'B', 'C'], weights=[2, 3, 4], values=[3, 4, 5], capacity=5'
    :return: A string describing the selected items and total value
    """
    try:
        # Extract parameters from the input string
        params = eval(f"dict({problem_input})")
        items = params['items']
        weights = params['weights']
        values = params['values']
        capacity = params['capacity']
        
        n = len(items)
        dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                if weights[i-1] <= w:
                    dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
                else:
                    dp[i][w] = dp[i-1][w]
        
        w = capacity
        selected_items = []
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                selected_items.append(items[i-1])
                w -= weights[i-1]
        
        return f"Selected items: {selected_items[::-1]}, Total value: {dp[n][capacity]}"
    except Exception as e:
        return f"Error solving knapsack problem: {str(e)}"

@tool
def fibonacci(n: str) -> str:
    """
    Finds the nth Fibonacci number.
    :param n: A string containing the position in the Fibonacci sequence (e.g., "10")
    :return: The nth Fibonacci number as a string
    """
    try:
        n = int(n)
        if n <= 1:
            return str(n)
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return f"The {n}th Fibonacci number is {b}"
    except Exception as e:
        return f"Error calculating Fibonacci number: {str(e)}"

# Collect the tools
tools = [solve_knapsack, fibonacci]

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a helpful AI assistant that solves problems step by step.
    For the knapsack problem, use the solve_knapsack tool with the parameters exactly as given.
    For the Fibonacci sequence, use the fibonacci tool with just the number.
    Always execute both tasks and combine their results."""),
    HumanMessage(content="""Available tools:
    solve_knapsack: Pass the parameters as a single string with items, weights, values, and capacity.
    fibonacci: Pass the position number as a string.
    
    {tools}"""),
    HumanMessage(content="{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Set up the agent
llm = ChatOpenAI(temperature=0, model="gpt-4")
agent = create_openai_tools_agent(llm, tools, prompt)

# Create an agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Example usage
if __name__ == "__main__":
    # Test direct tool usage first
    print("Direct tool testing:")
    knapsack_input = "items=['A', 'B', 'C'], weights=[2, 3, 4], values=[3, 4, 5], capacity=5"
    print("Knapsack result:", solve_knapsack(knapsack_input))
    print("Fibonacci result:", fibonacci("10"))
    
    print("\nTesting with agent:")
    query = ("Please solve these two problems:\n"
             "1. Solve the knapsack problem with these parameters:\n"
             "items=['A', 'B', 'C'], weights=[2, 3, 4], values=[3, 4, 5], capacity=5\n"
             "2. Find the 10th Fibonacci number")
    
    result = agent_executor.invoke({"input": query})
    print("\nFinal Result:", result)