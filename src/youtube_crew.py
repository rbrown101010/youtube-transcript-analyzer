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
    
    @agent
    def content_summarizer(self) -> Agent:
        return Agent(
            role="Content Summarizer",
            goal="Create concise, engaging summaries of the analyzed content",
            backstory="""You are a skilled content summarizer with a talent for distilling complex information 
                     into clear, engaging summaries. You excel at identifying the most important points and 
                     presenting them in a way that resonates with the audience.""",
            llm=ChatOpenAI(
                temperature=0.7,
                model_name="gpt-4"
            ),
            verbose=True
        )
    
    @agent
    def sentiment_analyzer(self) -> Agent:
        return Agent(
            role="Sentiment Analyzer",
            goal="Analyze the emotional tone and sentiment patterns in the content",
            backstory="""You are an expert in sentiment analysis with a deep understanding of emotional 
                     intelligence and language patterns. You can identify subtle emotional undertones 
                     and overall sentiment trends in content.""",
            llm=ChatOpenAI(
                temperature=0.5,
                model_name="gpt-4"
            ),
            verbose=True
        )
    
    @task
    def extract_transcript_task(self) -> Task:
        return Task(
            description="""Extract the transcript from the YouTube video and prepare it for analysis.""",
            expected_output="The complete transcript of the video ready for analysis.",
            agent=self.youtube_analyst()
        )
    
    @task
    def analyze_sentiment_task(self) -> Task:
        return Task(
            description="""Analyze the transcript for:
                       - Overall emotional tone
                       - Key sentiment patterns
                       - Emotional transitions
                       - Notable expressions and emphasis""",
            expected_output="A detailed sentiment analysis report of the content.",
            agent=self.sentiment_analyzer()
        )
    
    @task
    def create_summary_task(self) -> Task:
        return Task(
            description="""Create a comprehensive summary that includes:
                       - Main topics and key points
                       - Important insights
                       - Key takeaways
                       - Notable quotes or moments""",
            expected_output="An engaging and informative summary of the video content.",
            agent=self.content_summarizer()
        )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.youtube_analyst(),
                self.content_summarizer(),
                self.sentiment_analyzer()
            ],
            tasks=[
                self.extract_transcript_task(),
                self.analyze_sentiment_task(),
                self.create_summary_task()
            ],
            process=Process.sequential,
            verbose=True
        ) 