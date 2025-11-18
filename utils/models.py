from pydantic import BaseModel, Field
from typing import List

# The 7 categories from your database/db.py 'apps' table
APP_CATEGORIES = "('Songs', 'Entertainment', 'SocialMedia', 'Games', 'Communication', 'Help', 'Other')"

class AppRecommendation(BaseModel):
    app_name: str = Field(description="Name of recommended application")
    app_url: str = Field(description="URL or local path of the application")
    search_query: str = Field(description="Search query if web-based application")
    is_local: bool = Field(default=False, description="Whether the app is a local executable")
    category: str = Field(description=f"Category of the app, must be one of: {APP_CATEGORIES}")

class RecommendationResponse(BaseModel):
    recommendation: str = Field(description="4-word mood improvement suggestions")
    recommendation_options: List[AppRecommendation] = Field(description="Two app recommendations")

class RecommendationList(BaseModel):
    listofRecommendations: List[RecommendationResponse] = Field(description="List of 3 recommendations with options")