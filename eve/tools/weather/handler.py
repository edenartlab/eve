import json

import httpx

from eve.tool import ToolContext


async def handler(context: ToolContext):
    lat = context.args["lat"]
    lon = context.args["lon"]

    points_url = f"https://api.weather.gov/points/{lat},{lon}"
    # Provide a descriptive User-Agent per NOAA policy
    headers = {"User-Agent": "MyForecastApp (contact@example.com)"}

    async with httpx.AsyncClient() as client:
        # Step 1: Get the forecast endpoint
        points_resp = await client.get(points_url, headers=headers, timeout=30.0)
        points_data = points_resp.json()

        # Step 2: Use the "forecast" or "forecastHourly" property to get actual data
        forecast_url = points_data["properties"]["forecast"]
        forecast_resp = await client.get(forecast_url, headers=headers, timeout=30.0)
        forecast_data = forecast_resp.json()

    output = forecast_data["properties"]

    return {"output": json.dumps(output)}
