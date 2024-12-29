from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task, crew
from crewai_tools import SerperDevTool, WikipediaTools

@CrewBase
class AIResearchCrew:
    """Crew for researching and writing about AI technology"""
    
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    @agent
    def research_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['research_analyst'],
            tools=[SerperDevTool(), WikipediaTools()],
            verbose=True
        )
    
    @agent
    def content_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['content_writer'],
            verbose=True
        )
    
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task']
        )
    
    @task
    def writing_task(self) -> Task:
        return Task(
            config=self.tasks_config['writing_task']
        )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        ) 