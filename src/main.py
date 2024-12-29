from youtube_analysis_crew import YouTubeAnalysisCrew

def main():
    # Create and kickoff the crew
    crew = YouTubeAnalysisCrew()
    
    # You can pass the YouTube URL as an input
    result = crew.crew().kickoff(
        inputs={
            'youtube_url': 'https://www.youtube.com/watch?v=your-video-id'
        }
    )
    
    # Print the results
    print("\nAnalysis Results:")
    print(result)

if __name__ == "__main__":
    main() 