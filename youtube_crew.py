from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task, crew
from tools.youtube_tools import YouTubeTranscriptTool
from langchain.chat_models import ChatOpenAI

@CrewBase
class YouTubeAnalysisCrew:
    """Crew for analyzing YouTube video transcripts"""
    
    @agent
    def youtube_analyst(self) -> Agent:
        return Agent(
            role="YouTube Content Analyst",
            goal="Analyze YouTube video transcripts and provide comprehensive insights using OpenAI's capabilities",
            backstory="""You are an expert content analyst specializing in extracting meaningful insights from video content.
                     With your deep understanding of context and natural language processing, you excel at identifying
                     key themes, summarizing content, and providing valuable analysis of video transcripts.""",
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
            description="""Extract and analyze the transcript from a YouTube video to provide:
                       - Main topics and key points
                       - Sentiment analysis
                       - Key insights and takeaways
                       - Summary of the content""",
            expected_output="A detailed analysis report including the main points, insights, and overall summary of the video content.",
            agent=self.youtube_analyst()
        )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.youtube_analyst()],
            tasks=[self.analyze_video_task()],
            process=Process.sequential,
            verbose=True
        ) 