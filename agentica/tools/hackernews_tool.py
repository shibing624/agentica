# -*- encoding: utf-8 -*-
"""
@author: orange-crow, XuMing(xuming624@qq.com)
@description:
part of the code is from phidata
"""
import asyncio
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

    async def get_top_hackernews_stories(self, num_stories: int = 10) -> str:
        """Get top stories from Hacker News.

        Args:
            num_stories (int): Number of stories to return. Defaults to 10.

        Returns:
            str: JSON string of top stories.
        """

        logger.info(f"Getting top {num_stories} stories from Hacker News")
        async with httpx.AsyncClient() as client:
            # Fetch top story IDs
            response = await client.get("https://hacker-news.firebaseio.com/v0/topstories.json")
            story_ids = response.json()

            # Fetch story details in parallel
            async def fetch_story(story_id: int) -> dict:
                resp = await client.get(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json")
                story = resp.json()
                story["username"] = story["by"]
                return story

            tasks = [fetch_story(sid) for sid in story_ids[:num_stories]]
            stories = await asyncio.gather(*tasks)

        return json.dumps(list(stories), indent=2, ensure_ascii=False)

    async def get_user_details(self, username: str) -> str:
        """Get the details of a Hacker News user using their username.

        Args:
            username (str): Username of the user to get details for.

        Returns:
            str: JSON string of the user details.
        """

        try:
            logger.info(f"Getting details for user: {username}")
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"https://hacker-news.firebaseio.com/v0/user/{username}.json")
                user = resp.json()
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
    import asyncio

    m = HackerNewsTool()
    r = asyncio.run(m.get_top_hackernews_stories(3))
    print(r)
    r = asyncio.run(m.get_user_details('tomthe'))
    print(r)
