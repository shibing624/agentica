# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: weather tool
"""
import json
import requests
from agentica.tools.base import Tool
from agentica.utils.log import logger


class WeatherTool(Tool):
    def __init__(self):
        super().__init__(name="get_weather_tool")
        self.register(self.get_weather)

    def get_weather(self, city: str = None) -> str:
        """Get weather info for a city.

        Args:
            city (str): Name of the city, can be in Chinese or English, eg: "Beijing", "武汉" or "wuhan".
            if None, will get the weather for the current location.

        Returns:
            str: string representation of the weather data. Markdown format.
        """
        logger.info(f"get_current_weather({city})")
        try:
            endpoint = "https://wttr.in"
            if city:
                response = requests.get(f"{endpoint}/{city}")
            else:
                response = requests.get(endpoint)
            result = response.text

            logger.debug(f"Weather data for {city}: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in getting weather for {city}: {str(e)}")
            return json.dumps({"operation": "get_current_weather", "error": str(e)})


if __name__ == '__main__':
    tool = WeatherTool()
    print(tool.get_weather("Beijing"))
    print(tool.get_weather("武汉"))
    print(tool.get_weather())
