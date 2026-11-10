from langchain_core.tools import tool
from langchain.agents import create_agent
import requests
import json

from utils.models import model

@tool
def get_weather(latitude: float, longitude: float) -> str:
    """Get current temperature in Fahrenheit and weather code for given coordinates.

    Args:
        latitude: Latitude coordinate (-90 to 90)
        longitude: Longitude coordinate (-180 to 180)

    Returns:
        JSON string with temperature_fahrenheit and weather_code (do not include the code in your response, translate it to plain English)
    """
    if not (-90.0 <= latitude <= 90.0):
        return json.dumps({"error": f"Invalid latitude: {latitude}. Must be between -90 and 90."})
    if not (-180.0 <= longitude <= 180.0):
        return json.dumps({"error": f"Invalid longitude: {longitude}. Must be between -180 and 180."})

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,weather_code",
        "temperature_unit": "fahrenheit"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        current = data.get("current", {})
        temperature = current.get("temperature_2m")
        weather_code = current.get("weather_code")
        result = {
            "temperature_fahrenheit": temperature,
            "weather_code": weather_code
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"Failed to retrieve weather: {str(e)}"})

@tool
def get_user_preferences(user_id: str) -> str:
    """Get a user's saved preferences."""
    # Simulate a user database
    preferences = {
        "alice": "Loves sci-fi movies, prefers warm weather destinations",
        "bob": "Enjoys comedy films, likes cold climates for travel"
    }
    return preferences.get(user_id.lower(), "No preferences found")

@tool
def book_recommendation(genre: str, user_preferences: str = "") -> str:
    """Get personalized movie recommendations based on genre and user preferences."""
    recommendations = {
        "sci-fi": "Based on your preferences, try: Arrival, Ex Machina, or The Martian",
        "comedy": "Based on your preferences, try: The Big Lebowski, Anchorman, or Bridesmaids"
    }
    return recommendations.get(genre.lower(), "No recommendations available")

# Create a helpful assistant agent
agent = create_agent(
    model=model,
    tools=[get_weather, get_user_preferences, book_recommendation],
    system_prompt="""You are a helpful personal assistant. 
    
    You can:
    - Check weather for any city
    - Look up user preferences
    - Recommend movies based on preferences
    
    Always be friendly and personalize your responses based on user preferences."""
)
