from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task, crew
from tools.youtube_tools import YouTubeTranscriptTool
import os
from langchain.chat_models import ChatOpenAI

@CrewBase
class YouTubeAnalysisCrew:
    """Crew for analyzing YouTube video transcripts"""
    
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    @agent
    def youtube_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['youtube_analyst'],
            tools=[YouTubeTranscriptTool()],
            llm=ChatOpenAI(
                temperature=0.7,
                model_name="gpt-4"
            ),
            verbose=True
        )
    
    @task
    def analyze_video_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_video_task']
        )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        ) 