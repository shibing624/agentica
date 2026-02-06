#!/usr/bin/env python3
"""
天气查询工具 - 支持多种方式
"""

import requests
import json


def query_weather_wttr(city: str = "北京"):
    """使用 wttr.in 查询天气（备用方式）"""
    try:
        # 使用无 SSL 验证方式
        response = requests.get(
            f"http://wttr.in/{city}?format=j1",
            timeout=10
        )
        data = response.json()
        
        current = data['current_condition'][0]
        weather = data['weather'][0]
        
        return f"""
🌤️ {city} 天气预报

📍 当前天气:
   温度: {current['temp_C']}°C (体感 {current['FeelsLikeC']}°C)
   天气: {current['lang_zh'][0]['value'] if 'lang_zh' in current else current['weatherDesc'][0]['value']}
   湿度: {current['humidity']}%
   风速: {current['windspeedKmph']} km/h
   能见度: {current['visibility']} km
   更新时间: {current['observation_time']}

📅 今天 ({weather['date']}):
   最高: {weather['maxtempC']}°C
   最低: {weather['mintempC']}°C
   日出: {weather['astronomy'][0]['sunrise']}
   日落: {weather['astronomy'][0]['sunset']}
        """
    except Exception as e:
        return f"查询失败: {str(e)}"


def query_weather_simple(city: str = "北京"):
    """使用简化格式查询天气"""
    try:
        # 尝试使用简化格式
        response = requests.get(
            f"http://wttr.in/{city}?format=3",
            timeout=10
        )
        return f"🌤️ {city}: {response.text}"
    except Exception as e:
        return f"查询失败: {str(e)}"


if __name__ == "__main__":
    import sys
    
    # 获取城市名称，默认为北京
    city = sys.argv[1] if len(sys.argv) > 1 else "北京"
    
    print(f"🌤️ 正在查询 {city} 的天气...\n")
    
    # 首先尝试详细格式
    result = query_weather_wttr(city)
    if "查询失败" in result:
        # 如果失败，尝试简化格式
        result = query_weather_simple(city)
    
    print(result)