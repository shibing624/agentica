# -*- encoding: utf-8 -*-
"""
@author: orange-crow, XuMing(xuming624@qq.com)
@description:
part of the code is from phidata
"""
import json
import httpx
from agentica.tools.base import Tool
from agentica.utils.log import logger


class HackerNewsTool(Tool):
    def __init__(
        self,
        get_top_stories: bool = True,
        get_user_details: bool = True,
    ):
        super().__init__(name="hackers_news")

        # Register functions in the toolkit
        if get_top_stories:
            self.register(self.get_top_hackernews_stories)
        if get_user_details:
            self.register(self.get_user_details)

    def get_top_hackernews_stories(self, num_stories: int = 10) -> str:
        """Get top stories from Hacker News.

        Args:
            num_stories (int): Number of stories to return. Defaults to 10.

        Example:
            from agentica.tools.hackernews_tool import HackerNewsTool
            m = HackerNewsTool()
            top_stories = m.get_top_hackernews_stories(3)
            print(top_stories)

        Returns:
            str: JSON string of top stories.
        """

        logger.info(f"Getting top {num_stories} stories from Hacker News")
        # Fetch top story IDs
        response = httpx.get("https://hacker-news.firebaseio.com/v0/topstories.json")
        story_ids = response.json()

        # Fetch story details
        stories = []
        for story_id in story_ids[:num_stories]:
            story_response = httpx.get(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json")
            story = story_response.json()
            story["username"] = story["by"]
            stories.append(story)
        return json.dumps(stories, indent=2, ensure_ascii=False)

    def get_user_details(self, username: str) -> str:
        """Get the details of a Hacker News user using their username.

        Args:
            username (str): Username of the user to get details for.

        Example:
            from agentica.tools.hackernews_tool import HackerNewsTool
            m = HackerNewsTool()
            user_details = m.get_user_details('tomthe')
            print(user_details)

        Returns:
            str: JSON string of the user details.
        """

        try:
            logger.info(f"Getting details for user: {username}")
            user = httpx.get(f"https://hacker-news.firebaseio.com/v0/user/{username}.json").json()
            user_details = {
                "id": user.get("user_id"),
                "karma": user.get("karma"),
                "about": user.get("about"),
                "total_items_submitted": len(user.get("submitted", [])),
            }
            return json.dumps(user_details, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.exception(e)
            return f"Error getting user details: {e}"


if __name__ == '__main__':
    m = HackerNewsTool()
    r = m.get_top_hackernews_stories(3)
    print(r)
    r = m.get_user_details('tomthe')
    print(r)