# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Weather tool with fallback strategy
"""
import json
import os
import httpx
from agentica.tools.base import Tool
from agentica.utils.log import logger


class WeatherTool(Tool):
    """Weather tool with multiple fallback sources."""

    def __init__(self):
        super().__init__(name="get_weather_tool")
        self.register(self.get_weather)
        # API key for OpenWeatherMap (optional, falls back to free sources)
        self.openweather_api_key = os.getenv("OPENWEATHER_API_KEY")

    async def get_weather(self, city: str = None) -> str:
        """Get weather info for a city with fallback strategy.

        Args:
            city (str): Name of the city, can be in Chinese or English, eg: "Beijing", "武汉" or "wuhan".
            If None, will get the weather for the current location.

        Returns:
            str: String representation of the weather data in markdown format.
        """
        if not city:
            city = "auto:ip"

        # Try multiple sources with fallback
        errors = []

        # Source 1: OpenWeatherMap (if API key available)
        if self.openweather_api_key:
            try:
                result = await self._get_openweather(city)
                if result and "error" not in result:
                    logger.debug(f"OpenWeatherMap succeeded for '{city}'")
                    return result
                else:
                    errors.append(f"OpenWeatherMap: returned empty or error result")
            except Exception as e:
                logger.debug(f"OpenWeatherMap failed for '{city}': {e}")
                errors.append(f"OpenWeatherMap: {e}")
        else:
            logger.debug(f"OpenWeatherMap skipped for '{city}': no API key")

        # Source 2: wttr.in (with shorter timeout)
        try:
            result = await self._get_wttr(city)
            if result and "error" not in str(result).lower():
                logger.debug(f"wttr.in succeeded for '{city}'")
                return result
            else:
                errors.append(f"wttr.in: returned empty or error result")
        except Exception as e:
            logger.debug(f"wttr.in failed for '{city}': {e}")
            errors.append(f"wttr.in: {e}")

        # Source 3: Open-Meteo (free, no API key needed)
        try:
            result = await self._get_openmeteo(city)
            if result:
                logger.debug(f"Open-Meteo succeeded for '{city}'")
                return result
            else:
                errors.append(f"Open-Meteo: geocoding failed or returned no results")
        except Exception as e:
            logger.debug(f"Open-Meteo failed for '{city}': {e}")
            errors.append(f"Open-Meteo: {e}")

        # All sources failed
        error_msg = f"All weather sources failed for '{city}'. Errors: {'; '.join(errors)}"
        logger.warning(error_msg)
        return json.dumps({
            "operation": "get_weather",
            "city": city,
            "error": error_msg,
            "suggestion": "Please try again later or set OPENWEATHER_API_KEY for more reliable service."
        })

    async def _get_wttr(self, city: str) -> str:
        """Get weather from wttr.in (with timeout)."""
        endpoint = "https://wttr.in"
        url = f"{endpoint}/{city}" if city != "auto:ip" else endpoint

        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=5)
            if response.status_code == 200:
                return response.text
        return None

    async def _get_openweather(self, city: str) -> str:
        """Get weather from OpenWeatherMap (requires API key)."""
        if not self.openweather_api_key:
            return None

        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": self.openweather_api_key,
            "units": "metric",
            "lang": "en"
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=8)
            if response.status_code == 200:
                data = response.json()
                return self._format_openweather(data)
        return None

    def _format_openweather(self, data: dict) -> str:
        """Format OpenWeatherMap response to markdown."""
        city = data.get("name", "Unknown")
        country = data.get("sys", {}).get("country", "")
        weather = data.get("weather", [{}])[0]
        main = data.get("main", {})
        wind = data.get("wind", {})

        desc = weather.get("description", "Unknown").capitalize()
        temp = main.get("temp", "N/A")
        feels_like = main.get("feels_like", "N/A")
        humidity = main.get("humidity", "N/A")
        pressure = main.get("pressure", "N/A")
        wind_speed = wind.get("speed", "N/A")

        return f"""# Weather in {city}, {country}

**Condition:** {desc}
**Temperature:** {temp}°C (feels like {feels_like}°C)
**Humidity:** {humidity}%
**Pressure:** {pressure} hPa
**Wind Speed:** {wind_speed} m/s

_Source: OpenWeatherMap_
"""

    async def _get_openmeteo(self, city: str) -> str:
        """Get weather from Open-Meteo (free, no API key)."""
        # Common Chinese city name mappings
        city_mappings = {
            "北京": "Beijing",
            "上海": "Shanghai",
            "广州": "Guangzhou",
            "深圳": "Shenzhen",
            "杭州": "Hangzhou",
            "南京": "Nanjing",
            "武汉": "Wuhan",
            "成都": "Chengdu",
            "西安": "Xian",
            "重庆": "Chongqing",
            "天津": "Tianjin",
            "苏州": "Suzhou",
        }

        # Try English name if Chinese city name fails
        search_city = city_mappings.get(city, city)

        async with httpx.AsyncClient() as client:
            # First, geocode the city name to coordinates
            geo_url = "https://geocoding-api.open-meteo.com/v1/search"
            geo_params = {"name": search_city, "count": 1}

            geo_response = await client.get(geo_url, params=geo_params, timeout=8)
            if geo_response.status_code != 200:
                logger.debug(f"Open-Meteo geocoding failed for '{city}' (searched as '{search_city}'): HTTP {geo_response.status_code}")
                return None

            geo_data = geo_response.json()
            results = geo_data.get("results", [])
            if not results:
                logger.debug(f"Open-Meteo geocoding returned no results for '{city}' (searched as '{search_city}')")
                return None

            lat = results[0].get("latitude")
            lon = results[0].get("longitude")
            city_name = results[0].get("name", city)
            country = results[0].get("country", "")

            # Get weather data
            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_params = {
                "latitude": lat,
                "longitude": lon,
                "current": ["temperature_2m", "relative_humidity_2m", "weather_code", "wind_speed_10m", "pressure_msl"]
            }

            weather_response = await client.get(weather_url, params=weather_params, timeout=5)
            if weather_response.status_code != 200:
                return None

            data = weather_response.json()
            logger.debug(f"Open-Meteo get weather from {city}, response: {data}")
            current = data.get("current", {})

            return self._format_openmeteo(city_name, country, current)

    def _format_openmeteo(self, city: str, country: str, current: dict) -> str:
        """Format Open-Meteo response to markdown."""
        temp = current.get("temperature_2m", "N/A")
        humidity = current.get("relative_humidity_2m", "N/A")
        wind_speed = current.get("wind_speed_10m", "N/A")
        pressure = current.get("pressure_msl", "N/A")
        weather_code = current.get("weather_code", 0)

        # Simple weather code mapping
        weather_desc = self._get_weather_description(weather_code)

        return f"""# Weather in {city}, {country}

**Condition:** {weather_desc}
**Temperature:** {temp}°C
**Humidity:** {humidity}%
**Pressure:** {pressure} hPa
**Wind Speed:** {wind_speed} km/h

_Source: Open-Meteo_
"""

    def _get_weather_description(self, code: int) -> str:
        """Get weather description from WMO weather code."""
        codes = {
            0: "Clear sky",
            1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            56: "Light freezing drizzle", 57: "Dense freezing drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            66: "Light freezing rain", 67: "Heavy freezing rain",
            71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
            77: "Snow grains",
            80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
            85: "Slight snow showers", 86: "Heavy snow showers",
            95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
        }
        return codes.get(code, "Unknown")


if __name__ == '__main__':
    import asyncio

    tool = WeatherTool()
    print("=== Beijing ===")
    print(asyncio.run(tool.get_weather("Beijing")))
    print("\n=== Wuhan ===")
    print(asyncio.run(tool.get_weather("武汉")))
    print("\n=== Current Location ===")
    print(asyncio.run(tool.get_weather()))
