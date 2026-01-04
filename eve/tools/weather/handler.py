import asyncio
import json

import requests

from eve.tool import ToolContext


async def handler(context: ToolContext):
    lat = context.args["lat"]
    lon = context.args["lon"]

    points_url = f"https://api.weather.gov/points/{lat},{lon}"
    # Provide a descriptive User-Agent per NOAA policy
    headers = {"User-Agent": "MyForecastApp (contact@example.com)"}

    # Step 1: Get the forecast endpoint
    points_resp = await asyncio.to_thread(requests.get, points_url, headers=headers)
    points_data = points_resp.json()

    # Step 2: Use the "forecast" or "forecastHourly" property to get actual data
    forecast_url = points_data["properties"]["forecast"]
    forecast_resp = await asyncio.to_thread(requests.get, forecast_url, headers=headers)
    forecast_data = forecast_resp.json()

    output = forecast_data["properties"]

    return {"output": json.dumps(output)}
