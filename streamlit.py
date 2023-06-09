import langchain
from typing_extensions import Text
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import gpt4free
from gpt4free import Provider, forefront
import wikipedia
import wolframalpha



class freegpt(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if isinstance(stop, list):
            stop = stop + ["\n###","\nObservation:", "\nObservations:"]
            
        response = gpt4free.Completion.create(provider=Provider.UseLess, prompt=prompt)
        response = response['text']
        response = response.split("Observation", maxsplit=1)[0]

        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
from langchain import HuggingFaceHub
from langchain.llms import VertexAI
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
import re
import streamlit as st
# Define which tools the agent can use to answer user queries


def ai(input):
  try:
    response = gpt4free.Completion.create(provider=Provider.UseLess, prompt=input)
    response=response['text']
  except Exception as e:
    return "failed. please try again later."
  return response
def executor():
  st.header("Chat with Ava")
  st.write("please note it will respond but it does take long. if you see running at the top then it is working. you may experience errors. i am working on fixing it)
  search = SerpAPIWrapper(serpapi_api_key='cc528133d4712378d13ee296bb2965e4c9d511ab22bd7c8819bd61bdc9d66c9c')
  wiki = WikipediaAPIWrapper()
  wolf = WolframAlphaAPIWrapper(wolfram_alpha_appid='R94U89-4738P78QQ7')
  tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name = "Wikipedia",
        func=wiki.run,
        description="uses wikipedia to find useful information"

    ),
    Tool(
        name="Wolfram Alpha",
        func=wolf.run,
        description="use for simple to very complex calculations"
    ),
    Tool(
        name="AI Answerer",
        func=ai,
        description="uses ai to answer questions. can answer in any language and has limited knowledge from after 2021."
    )
]
  template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Always use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of the tools. It should just be the name of the tool(eg. Search). if no tool is selected then leave blank.
Action Input: the input to the action or tool chosen in Action.if none leave blank.
Observation: the result of the action.if none leave blank.
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! 


Question: {input}
{agent_scratchpad}"""

# Set up a prompt template
  class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

  prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)
  class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        print(llm_output)
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
  output_parser = CustomOutputParser()
# LLM chain consisting of the LLM and a prompt


  llm = freegpt()

  llm_chain = LLMChain(llm=llm, prompt=prompt)
  tool_names = [tool.name for tool in tools]
  agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)
  query = st.text_input("Ask questions about your PDF file:")
  agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
  if query:
    st.write(agent_executor.run(query))

if __name__ == '__main__':
  executor()
