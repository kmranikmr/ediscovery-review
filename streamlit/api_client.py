"""
API Client for the LLM Retrieval System.
This module provides utilities for calling API endpoints.
"""
import requests
import json
import streamlit as st
import os
from typing import Dict, Any, Optional

# Default API base URL
DEFAULT_API_URL = "http://localhost:8000"

def get_api_base_url() -> str:
    """
    Get the base URL for the API from environment variables or use default
    """
    return os.environ.get("API_BASE_URL", DEFAULT_API_URL)

def call_api_endpoint(endpoint: str, data: Dict[str, Any], method: str = "POST") -> Dict[str, Any]:
    """
    Call an API endpoint and return the response
    
    Args:
        endpoint: The API endpoint path (e.g., "/search")
        data: The data to send to the API
        method: The HTTP method to use (default: "POST")
        
    Returns:
        The API response as a dictionary
    """
    # Ensure endpoint starts with a slash
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"
    
    # Construct the full URL
    url = f"{get_api_base_url()}{endpoint}"
    
    try:
        # Show a small indication that we're making an API call
        with st.status(f"Calling API endpoint: {endpoint}", expanded=False) as status:
            # Make the API call
            if method.upper() == "POST":
                response = requests.post(url, json=data, timeout=120)  # 2-minute timeout
            elif method.upper() == "GET":
                response = requests.get(url, params=data, timeout=60)  # 1-minute timeout
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Check if the request was successful
            if response.status_code == 200:
                # Try to parse the response as JSON
                try:
                    result = response.json()
                    status.update(label="API call successful", state="complete")
                    return {"success": True, "result": result}
                except json.JSONDecodeError:
                    status.update(label="Failed to parse API response", state="error")
                    return {
                        "success": False,
                        "error": "Failed to parse API response as JSON",
                        "raw_response": response.text
                    }
            else:
                # Handle error responses
                status.update(label=f"API error: {response.status_code}", state="error")
                try:
                    error_data = response.json()
                    error_message = error_data.get("detail", f"API error: {response.status_code}")
                except:
                    error_message = f"API error: {response.status_code} - {response.text}"
                
                return {"success": False, "error": error_message, "status_code": response.status_code}
    
    except requests.exceptions.RequestException as e:
        # Handle connection errors
        return {"success": False, "error": f"Connection error: {str(e)}"}
    except Exception as e:
        # Handle any other errors
        return {"success": False, "error": f"Error: {str(e)}"}