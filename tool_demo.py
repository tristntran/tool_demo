from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, LLMChain
from langchain_community.llms import Replicate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import StructuredTool
from typing import List, Union, Tuple
import re

# Define tools using the @tool decorator
from langchain.tools import tool

@tool
def solve_knapsack(items: List[str], weights: List[int], values: List[int], capacity: int) -> Tuple[List[str], int]:
    """
    Solves the knapsack problem.
    :param items: List of item names
    :param weights: List of item weights
    :param values: List of item values
    :param capacity: Knapsack capacity
    :return: Tuple of (selected items, total value)
    """
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
    
    return selected_items[::-1], dp[n][capacity]

@tool
def fibonacci(n: int) -> int:
    """
    Finds the nth Fibonacci number.
    :param n: Position in the Fibonacci sequence
    :return: The nth Fibonacci number
    """
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Collect the tools
tools = [solve_knapsack, fibonacci]

# Define the prompt template
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[StructuredTool]
    
    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

prompt = CustomPromptTemplate(
    template="""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}""",
    tools=tools,
    input_variables=["input", "intermediate_steps"]
)

# Define the output parser
class CustomOutputParser:
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

# Set up the agent
llm = OpenAI(temperature=0, model = "gpt-4o-mini-2024-07-18")
# llm = Replicate(
#     model="meta/meta-llama-3-8b-instruct",
#     model_kwargs={"temperature": 0.75, "max_length": 500, "top_p": 1},
# ) # Use this line instead of the OpenAI line to use the MetaLlama model
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=CustomOutputParser(),
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)

# Set up the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

# Example usage
result = agent_executor.run("Solve a knapsack problem with items=['A', 'B', 'C'], weights=[2, 3, 4], values=[3, 4, 5], and capacity=5. Then find the 10th Fibonacci number.")
print(result)