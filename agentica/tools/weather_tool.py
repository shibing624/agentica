# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: weather tool
"""
import json
import requests
from agentica.tools.base import Toolkit
from agentica.utils.log import logger


class WeatherTool(Toolkit):
    def __init__(
            self,
            enable_get_current_weather=True,
    ):
        super().__init__(name="weather_tool")
        if enable_get_current_weather:
            self.register(self.get_current_weather)

    def get_current_weather(self, city: str) -> str:
        """Get current weather for a city.

        Args:
            city (str): Name of the city.

        Returns:
            str: string representation of the weather data. Markdown format.
        """
        logger.info(f"get_current_weather({city})")
        try:
            endpoint = "https://wttr.in"
            response = requests.get(f"{endpoint}/{city}")
            result = response.text

            logger.debug(f"Weather data for {city}: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in getting weather for {city}: {str(e)}")
            return json.dumps({"operation": "get_current_weather", "error": str(e)})


if __name__ == '__main__':
    tool = WeatherTool()
    print(tool.get_current_weather("Beijing"))
    print(tool.get_current_weather("武汉"))
