# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Media handling methods for Agent

This module contains methods for handling images and videos.
"""

from typing import (
    List,
    Optional,
    TYPE_CHECKING,
)

from agentica.model.content import Image, Video

if TYPE_CHECKING:
    from agentica.agent.base import Agent


class MediaMixin:
    """Mixin class containing media handling methods for Agent."""

    def add_image(self: "Agent", image: Image) -> None:
        """Add an image to the agent.
        
        Args:
            image: The Image object to add.
        """
        if self.images is None:
            self.images = []
        self.images.append(image)
        if self.run_response is not None:
            if self.run_response.images is None:
                self.run_response.images = []
            self.run_response.images.append(image)

    def add_video(self: "Agent", video: Video) -> None:
        """Add a video to the agent.
        
        Args:
            video: The Video object to add.
        """
        if self.videos is None:
            self.videos = []
        self.videos.append(video)
        if self.run_response is not None:
            if self.run_response.videos is None:
                self.run_response.videos = []
            self.run_response.videos.append(video)

    def get_images(self: "Agent") -> Optional[List[Image]]:
        """Get all images associated with this agent.
        
        Returns:
            List of Image objects, or None if no images.
        """
        return self.images

    def get_videos(self: "Agent") -> Optional[List[Video]]:
        """Get all videos associated with this agent.
        
        Returns:
            List of Video objects, or None if no videos.
        """
        return self.videos
