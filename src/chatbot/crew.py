import os
from dotenv import load_dotenv
from crewai import Crew, Agent, Task, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.llm import LLM

load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_llm = LLM(model="google/gemini-2.0-flash")

@CrewBase
class Chatbot():
    """Summarizer crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def query_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['query_agent'],
            llm=gemini_llm,
            verbose=True
        )
    
    @agent
    def db_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['db_agent'],
            llm=gemini_llm,
            verbose=True
        )

    @agent
    def summarizer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['summarizer_agent'],
            llm=gemini_llm,
            verbose=True
        )

    @task
    def refine_query(self) -> Task:
        return Task(
            config=self.tasks_config['refine_query'],
        )
    
    @task
    def search_database(self) -> Task:
        def run_search(context):
            refined = context.get("refine_query") or {}
            refined_query = refined.get("refined_query") or ""

            original_query = context.get("user_query") or context.get("inputs", {}).get("user_query") or ""

            from tools.tools_utils import get_relevant_ids_booleans, fetch_docs
            
            ids = get_relevant_ids_booleans(original_query, refined_query, min_should=1)
            docs_map = fetch_docs(ids) if ids else {}
            docs = list(docs_map.values())
            return {"ids": ids, "docs": docs}

        return Task(
            config=self.tasks_config['search_database'],
            agent=None,
            context=[self.refine_query()],
            function=run_search,
            # use_llm=False
        )
    
    @task
    def summarize_answers(self) -> Task:
        return Task(
            config=self.tasks_config['summarize_answers'],
        )


    @crew
    def crew(self) -> Crew:
        """Creates the Chatbot crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            tracing=False
        )

    def kickoff(self, inputs):
        """
        Run the crew with the provided inputs.
        This method ensures inputs are properly passed to the crew.
        """
        return self.crew().kickoff(inputs=inputs)
