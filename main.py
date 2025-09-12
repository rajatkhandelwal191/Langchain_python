from dotenv import load_dotenv
load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

tools = [TavilySearch()]
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

chain = agent_executor



def main():
    print("Hello from search-agent!")
    result = chain.invoke(
        input={
            "input":"search for 3 job postings for ai engineer using langchain in the bay area in linkedin and list their details "
        }
    )
    print(result)


if __name__ == "__main__":
    main()
