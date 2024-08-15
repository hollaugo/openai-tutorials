from openai import OpenAI
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime
import json
import os
import time
import pandas as pd
import requests
from pytrends.request import TrendReq
from enum import Enum
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI()
MODEL = "gpt-4o-2024-08-06"

# Define Enums for Google properties
class GpropEnum(str, Enum):
    web = ''
    images = 'images'
    news = 'news'
    youtube = 'youtube'
    froogle = 'froogle'

# Define Data Model for Keyword Analysis Parameters
class KeywordAnalysisParams(BaseModel):
    kw_list: List[str]
    timeframe: str = 'today 5-y'
    geo: str = ''
    gprop: GpropEnum = GpropEnum.web

class KeywordReportOutput(BaseModel):
    summary:str
    suggestions: List[str]
    content_ideas: List[str]



KeywordReportOutput.model_rebuild() 

# Function to convert related topics DataFrame to a simplified list of objects with title and formatted value
def extract_topic_titles(related_topics_df):
    if related_topics_df is not None:
        return [{"topic_title": row["topic_title"], "formattedValue": row["formattedValue"], "value": row["value"]} for _, row in related_topics_df.iterrows()]
    return []

# Function to get Google Trends data with error handling and extended output
def get_detailed_analysis_of_keyword(params: KeywordAnalysisParams) -> Dict[str, Any]:
    result = {"related_topics": {}, "summary": {}, "suggestions": []}
    
    try:
        # Log the parameters
        print(f"Params: kw_list={params.kw_list}, timeframe={params.timeframe}, geo={params.geo}, gprop={params.gprop}")
        
        time.sleep(2)  # Handle rate limits
        pytrends = TrendReq()
        pytrends.build_payload(params.kw_list, cat=0, timeframe=params.timeframe, geo=params.geo, gprop=params.gprop)
        df = pytrends.interest_over_time().fillna(False)

        if df.empty:
            print("No data returned for the given parameters.")
            return result

        for keyword in params.kw_list:
            if keyword in df.columns:
                mean_value = df[keyword].mean()
                median_value = df[keyword].median()
                std_dev = df[keyword].std()
                max_interest_date = df[keyword].idxmax()
                min_interest_date = df[keyword].idxmin()

                result["summary"][keyword] = {
                    "mean": float(mean_value),
                    "median": float(median_value),
                    "std_dev": float(std_dev),
                    "max_interest_date": str(max_interest_date),
                    "max_interest": int(df[keyword].max()),
                    "min_interest_date": str(min_interest_date),
                    "min_interest": int(df[keyword].min())
                }

                related_topics = pytrends.related_topics()
                if related_topics and keyword in related_topics:
                    top_related = related_topics[keyword]['top']
                    rising_related = related_topics[keyword]['rising']
                    
                    result["related_topics"][keyword] = {
                        "top": extract_topic_titles(top_related),
                        "rising": extract_topic_titles(rising_related),
                        "breakout": extract_topic_titles(rising_related[rising_related['formattedValue'] == 'Breakout'])
                    }

                # Get suggestions
                suggestions = pytrends.suggestions(keyword=keyword)
                result["suggestions"].extend(suggestions)
                
            else:
                print(f"Keyword '{keyword}' not found in the data.")

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
    except Exception as e:
        print(f"An error occurred during the analysis: {e}")
    print(result)
    return result

# Define the keyword analysis function as a tool
keyword_analysis_function = {
    "type": "function",
    "function": {
        "name": "get_keyword_analysis",
        "description": "Analyze keywords using Google Trends based on user input",
        "parameters": {
            "type": "object",
            "properties": {
                "kw_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of keywords to analyze"
                },
                "timeframe": {
                    "type": "string",
                    "description": "The time range over which to analyze the keywords"
                },
                "geo": {
                    "type": "string",
                    "description": "The geographical location for the analysis"
                },
                "gprop": {
                    "type": "string",
                    "description": "The Google property to focus on",
                    "enum": ["", "images", "news", "youtube", "froogle"]
                }
            },
            "required": ["kw_list", "timeframe", "geo", "gprop"],
            "additionalProperties": False
        },
        "strict": True
    }
}

# Function to handle the tool call and execute it
def execute_tool(tool_call):
    # Accessing the arguments from the tool_call object
    tool_arguments = json.loads(tool_call.function.arguments)
    
    # Create a KeywordAnalysisParams object from the extracted arguments
    params = KeywordAnalysisParams(
        kw_list=tool_arguments['kw_list'],
        timeframe=tool_arguments['timeframe'],
        geo=tool_arguments['geo'],
        gprop=GpropEnum(tool_arguments['gprop'])
    )
    
    # Perform the keyword analysis
    analysis_result = get_detailed_analysis_of_keyword(params)
    
    return analysis_result

def handle_keyword_analysis(query, client):
    # Step 1: Initiate the tool call
    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a marketing research assistant tasked with analyzing keyword trends. Review the provided data and generate a report recommending the best content strategy for creating SEO-optimized content on trending topics."},
            {"role": "user", "content": query}
        ],
        tools=[keyword_analysis_function],
        response_format=KeywordReportOutput
    )
    
    # Step 2: Execute the tool call
    tool_call = response.choices[0].message.tool_calls[0]
    analysis_result = execute_tool(tool_call)
    
    # Step 3: Integrate the result back into the assistant's response (without using the 'tool' role)
    response_with_tool_result = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "user", "content": query},
            {"role": "assistant", "content": json.dumps(analysis_result, indent=2)}
        ],
        response_format=KeywordReportOutput
    )
    
    # Extract the clean JSON output from the response
    final_content = response_with_tool_result.choices[0].message.parsed

    # Return the final clean JSON result
    return final_content

# Example usage
query = "Analyze topics related to ai automation agency over the past year in the United States."
result = handle_keyword_analysis(query, client)

# Convert the Pydantic model to a dictionary and print it as clean JSON
print(json.dumps(result.dict(), indent=4))


