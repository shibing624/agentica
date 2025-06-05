# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
from __future__ import annotations
import datetime
import asyncio
import io
import json
import os
import random
import re
import urllib.parse
from copy import deepcopy
from typing import (
    Any,
    BinaryIO,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
    cast,
)

from PIL import Image, ImageDraw, ImageFont

from agentica.tools.video_analysis_tool import VideoAnalysisTool
from agentica.model.base import Model
from agentica.model.message import UserMessage
from agentica.model.openai.chat import OpenAIChat
from agentica.agent import Agent
from agentica.tools.base import Tool
from agentica.utils.log import logger
from agentica.config import AGENTICA_HOME
from agentica.utils.filename import sanitize_filename

# Constants
TOP_NO_LABEL_ZONE = 20

WEB_AGENT_SYSTEM_PROMPT = """
You are a helpful web agent that can assist users in browsing the web.
Given a high-level task, you can leverage predefined browser tools to help
users achieve their goals.
        """

PLANNING_AGENT_SYSTEM_PROMPT = """
You are a helpful planning agent that can assist users in planning complex
tasks which need multi-step browser interaction.
        """

OBSERVE_PROMPT_TEMPLATE = """
Please act as a web agent to help me complete the following high-level task:
<task>{task_prompt}</task>
Now, I have made screenshot (only the current viewport, not the full webpage)
based on the current browser state, and marked interactive elements in the
webpage.
Please carefully examine the requirements of the task, and current state of
the browser, and provide the next appropriate action to take.

{detailed_plan_prompt}

Here are the current available browser functions you can use:
{AVAILABLE_ACTIONS_PROMPT}

Here are the latest {history_window} trajectory (at most) you have taken:
<history>
{history}
</history>

Your output should be in json format, including the following fields:
- `observation`: The detailed image description about the current viewport. Do
not over-confident about the correctness of the history actions. You should
always check the current viewport to make sure the correctness of the next
action.
- `reasoning`: The reasoning about the next action you want to take, and the
possible obstacles you may encounter, and how to solve them. Do not forget to
check the history actions to avoid the same mistakes.
- `action_code`: The action code you want to take. It is only one step action
code, without any other texts (such as annotation)

Here is two example of the output:
```json
{{
    "observation": [IMAGE_DESCRIPTION],
    "reasoning": [YOUR_REASONING],
    "action_code": "fill_input_id([ID], [TEXT])"
}}

{{
    "observation":  "The current page is a CAPTCHA verification page on Amazon. It asks the user to ..",
    "reasoning": "To proceed with the task of searching for products, I need to complete..",
    "action_code": "fill_input_id(3, 'AUXPMR')"
}}

Here are some tips for you:
- Never forget the overall question: **{task_prompt}**
- Maybe after a certain operation (e.g. click_id), the page content has not
changed. You can check whether the action step is successful by looking at the
`success` of the action step in the history. If successful, it means that the
page content is indeed the same after the click. You need to try other methods.
- If using one way to solve the problem is not successful, try other ways.
Make sure your provided ID is correct!
- Some cases are very complex and need to be achieve by an iterative process.
You can use the `back()` function to go back to the previous page to try other
methods.
- There are many links on the page, which may be useful for solving the
problem. You can use the `click_id()` function to click on the link to see if
it is useful.
- Always keep in mind that your action must be based on the ID shown in the
current image or viewport, not the ID shown in the history.
- Do not use `stop()` lightly. Always remind yourself that the image only
shows a part of the full page. If you cannot find the answer, try to use
functions like `scroll_up()` and `scroll_down()` to check the full content of
the webpage before doing anything else, because the answer or next key step
may be hidden in the content below.
- If the webpage needs human verification, you must avoid processing it.
Please use `back()` to go back to the previous page, and try other ways.
- If you have tried everything and still cannot resolve the issue, please stop
the simulation, and report issues you have encountered.
- Check the history actions carefully, detect whether you have repeatedly made
the same actions or not.
- When dealing with wikipedia revision history related tasks, you need to
think about the solution flexibly. First, adjust the browsing history
displayed on a single page to the maximum, and then make use of the
find_text_on_page function. This is extremely useful which can quickly locate
the text you want to find and skip massive amount of useless information.
- Flexibly use interactive elements like slide down selection bar to filter
out the information you need. Sometimes they are extremely useful.
```
"""  # noqa: E501

GET_FINAL_ANSWER_PROMPT_TEMPLATE = """
We are solving a complex web task which needs multi-step browser interaction. After the multi-step observation, reasoning and acting with web browser, we think that the task is currently solved.
Here are all trajectory we have taken:
<history>{history}</history>
Please find the final answer, or give valuable insights and founds (e.g. if previous actions contain downloading files, your output should include the path of the downloaded file) about the overall task: <task>{task_prompt}</task>
        """  # noqa: E501

TASK_PLANNING_PROMPT_TEMPLATE = """
<task>{task_prompt}</task>
According to the problem above, if we use browser interaction, what is the general process of the interaction after visiting the webpage `{start_url}`? 

Please note that it can be viewed as Partially Observable MDP. Do not over-confident about your plan.
Please first restate the task in detail, and then provide a detailed plan to solve the task.
"""  # noqa: E501

TASK_REPLANNING_PROMPT_TEMPLATE = """
We are using browser interaction to solve a complex task which needs multi-step actions.
Here are the overall task:
<overall_task>{task_prompt}</overall_task>

In order to solve the task, we made a detailed plan previously. Here is the detailed plan:
<detailed plan>{detailed_plan}</detailed plan>

According to the task above, we have made a series of observations, reasonings, and actions. Here are the latest {history_window} trajectory (at most) we have taken:
<history>{history}</history>

However, the task is not completed yet. As the task is partially observable, we may need to replan the task based on the current state of the browser if necessary.
Now please carefully examine the current task planning schema, and our history actions, and then judge whether the task needs to be fundamentally replanned. If so, please provide a detailed replanned schema (including the restated overall task).

Your output should be in json format, including the following fields:
- `if_need_replan`: bool, A boolean value indicating whether the task needs to be fundamentally replanned.
- `replanned_schema`: str, The replanned schema for the task, which should not be changed too much compared with the original one. If the task does not need to be replanned, the value should be an empty string. 
"""  # noqa: E501

AVAILABLE_ACTIONS_PROMPT = """
1. `fill_input_id(identifier: Union[str, int], text: str)`: Fill an input
field (e.g. search box) with the given text and press Enter.
2. `click_id(identifier: Union[str, int])`: Click an element with the given ID.
3. `hover_id(identifier: Union[str, int])`: Hover over an element with the
given ID.
4. `download_file_id(identifier: Union[str, int])`: Download a file with the
given ID. It returns the path to the downloaded file. If the file is
successfully downloaded, you can stop the simulation and report the path to
the downloaded file for further processing.
5. `scroll_to_bottom()`: Scroll to the bottom of the page.
6. `scroll_to_top()`: Scroll to the top of the page.
7. `scroll_up()`: Scroll up the page. It is suitable when you want to see the
elements above the current viewport.
8. `scroll_down()`: Scroll down the page. It is suitable when you want to see
the elements below the current viewport. If the webpage does not change, It
means that the webpage has scrolled to the bottom.
9. `back()`: Navigate back to the previous page. This is useful when you want
to go back to the previous page, as current page is not useful.
10. `stop()`: Stop the action process, because the task is completed or failed
(impossible to find the answer). In this situation, you should provide your
answer in your output.
11. `get_url()`: Get the current URL of the current page.
12. `find_text_on_page(search_text: str)`: Find the next given text on the
current whole page, and scroll the page to the targeted text. It is equivalent
to pressing Ctrl + F and searching for the text, and is powerful when you want
to fast-check whether the current page contains some specific text.
13. `visit_page(url: str)`: Go to the specific url page.
14. `click_blank_area()`: Click a blank area of the page to unfocus the
current element. It is useful when you have clicked an element but it cannot
unfocus itself (e.g. Menu bar) to automatically render the updated webpage.
15. `ask_question_about_video(question: str)`: Ask a question about the
current webpage which contains video, e.g. youtube websites.
"""

ACTION_WITH_FEEDBACK_LIST = [
    'ask_question_about_video',
    'download_file_id',
    'find_text_on_page',
]


# TypedDicts
class DOMRectangle(TypedDict):
    x: Union[int, float]
    y: Union[int, float]
    width: Union[int, float]
    height: Union[int, float]
    top: Union[int, float]
    right: Union[int, float]
    bottom: Union[int, float]
    left: Union[int, float]


class VisualViewport(TypedDict):
    height: Union[int, float]
    width: Union[int, float]
    offsetLeft: Union[int, float]
    offsetTop: Union[int, float]
    pageLeft: Union[int, float]
    pageTop: Union[int, float]
    scale: Union[int, float]
    clientWidth: Union[int, float]
    clientHeight: Union[int, float]
    scrollWidth: Union[int, float]
    scrollHeight: Union[int, float]


class InteractiveRegion(TypedDict):
    tag_name: str
    role: str
    aria_name: str
    v_scrollable: bool
    rects: List[DOMRectangle]


# Helper Functions
def _get_str(d: Any, k: str) -> str:
    r"""Safely retrieve a string value from a dictionary."""
    if k not in d:
        raise KeyError(f"Missing required key: '{k}'")
    val = d[k]
    if isinstance(val, str):
        return val
    raise TypeError(
        f"Expected a string for key '{k}', but got {type(val).__name__}"
    )


def _get_number(d: Any, k: str) -> Union[int, float]:
    r"""Safely retrieve a number (int or float) from a dictionary"""
    val = d[k]
    if isinstance(val, (int, float)):
        return val
    raise TypeError(
        f"Expected a number (int/float) for key "
        f"'{k}', but got {type(val).__name__}"
    )


def _get_bool(d: Any, k: str) -> bool:
    r"""Safely retrieve a boolean value from a dictionary."""
    val = d[k]
    if isinstance(val, bool):
        return val
    raise TypeError(
        f"Expected a boolean for key '{k}', but got {type(val).__name__}"
    )


def _parse_json_output(
        text: str, logger: Any
) -> Dict[str, Any]:  # Added logger argument
    r"""Extract JSON output from a string."""

    markdown_pattern = r'```(?:json)?\\s*(.*?)\\s*```'
    markdown_match = re.search(markdown_pattern, text, re.DOTALL)
    if markdown_match:
        text = markdown_match.group(1).strip()

    triple_quotes_pattern = r'"""(?:json)?\\s*(.*?)\\s*"""'
    triple_quotes_match = re.search(triple_quotes_pattern, text, re.DOTALL)
    if triple_quotes_match:
        text = triple_quotes_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            # Attempt to fix common JSON issues like unquoted keys or using
            # single quotes
            # This is a simplified fix, more robust parsing might be needed
            # for complex cases
            fixed_text = re.sub(
                r"(\w+)(?=\s*:)", r'"\1"', text
            )  # Add quotes to keys
            fixed_text = fixed_text.replace("'", '"')  # Replace single quotes
            # with double
            # Handle boolean values not in lowercase
            fixed_text = re.sub(
                r':\s*True', ': true', fixed_text, flags=re.IGNORECASE
            )
            fixed_text = re.sub(
                r':\s*False', ': false', fixed_text, flags=re.IGNORECASE
            )
            # Remove trailing commas
            fixed_text = re.sub(r",\s*([\}\]])", r"\1", fixed_text)

            return json.loads(fixed_text)
        except json.JSONDecodeError:
            # Fallback to regex extraction if strict JSON parsing fails
            result = {}
            try:
                # Extract boolean-like values
                bool_pattern = r'"?(\w+)"?\s*:\s*(true|false)'
                for match in re.finditer(bool_pattern, text, re.IGNORECASE):
                    key, value = match.groups()
                    result[key.strip('"')] = value.lower() == "true"

                # Extract string values
                str_pattern = r'"?(\w+)"?\s*:\s*"([^"]*)"'
                for match in re.finditer(str_pattern, text):
                    key, value = match.groups()
                    result[key.strip('"')] = value

                # Extract numeric values
                num_pattern = r'"?(\w+)"?\s*:\s*(-?\d+(?:\.\d+)?)'
                for match in re.finditer(num_pattern, text):
                    key, value = match.groups()
                    try:
                        result[key.strip('"')] = int(value)
                    except ValueError:
                        result[key.strip('"')] = float(value)

                # Extract empty string values
                empty_str_pattern = r'"?(\w+)"?\s*:\s*""'
                for match in re.finditer(empty_str_pattern, text):
                    key = match.group(1)
                    result[key.strip('"')] = ""

                if result:
                    return result

                logger.warning(
                    f"Failed to parse JSON output after multiple attempts: "
                    f"{text}"
                )
                return {}
            except Exception as e:
                logger.warning(
                    f"Error during regex extraction from JSON-like string: {e}"
                )
                return {}


def _reload_image(image: Image.Image) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return Image.open(buffer)


def dom_rectangle_from_dict(rect: Dict[str, Any]) -> DOMRectangle:
    r"""Create a DOMRectangle object from a dictionary."""
    return DOMRectangle(
        x=_get_number(rect, "x"),
        y=_get_number(rect, "y"),
        width=_get_number(rect, "width"),
        height=_get_number(rect, "height"),
        top=_get_number(rect, "top"),
        right=_get_number(rect, "right"),
        bottom=_get_number(rect, "bottom"),
        left=_get_number(rect, "left"),
    )


def interactive_region_from_dict(region: Dict[str, Any]) -> InteractiveRegion:
    r"""Create an :class:`InteractiveRegion` object from a dictionary."""
    typed_rects: List[DOMRectangle] = []
    for rect_data in region[
        "rects"
    ]:  # Renamed rect to rect_data to avoid conflict
        typed_rects.append(dom_rectangle_from_dict(rect_data))

    return InteractiveRegion(
        tag_name=_get_str(region, "tag_name"),
        role=_get_str(region, "role"),
        aria_name=_get_str(region, "aria-name"),
        v_scrollable=_get_bool(region, "v-scrollable"),
        rects=typed_rects,
    )


def visual_viewport_from_dict(viewport: Dict[str, Any]) -> VisualViewport:
    r"""Create a :class:`VisualViewport` object from a dictionary."""
    return VisualViewport(
        height=_get_number(viewport, "height"),
        width=_get_number(viewport, "width"),
        offsetLeft=_get_number(viewport, "offsetLeft"),
        offsetTop=_get_number(viewport, "offsetTop"),
        pageLeft=_get_number(viewport, "pageLeft"),
        pageTop=_get_number(viewport, "pageTop"),
        scale=_get_number(viewport, "scale"),
        clientWidth=_get_number(viewport, "clientWidth"),
        clientHeight=_get_number(viewport, "clientHeight"),
        scrollWidth=_get_number(viewport, "scrollWidth"),
        scrollHeight=_get_number(viewport, "scrollHeight"),
    )


def add_set_of_mark(
        screenshot: Union[bytes, Image.Image, io.BufferedIOBase],
        ROIs: Dict[str, InteractiveRegion],
) -> Tuple[Image.Image, List[str], List[str], List[str]]:
    if isinstance(screenshot, Image.Image):
        return _add_set_of_mark(screenshot, ROIs)

    if isinstance(screenshot, bytes):
        screenshot = io.BytesIO(screenshot)

    image = Image.open(cast(BinaryIO, screenshot))
    comp, visible_rects, rects_above, rects_below = _add_set_of_mark(
        image, ROIs
    )
    image.close()
    return comp, visible_rects, rects_above, rects_below


def _add_set_of_mark(
        screenshot: Image.Image, ROIs: Dict[str, InteractiveRegion]
) -> Tuple[Image.Image, List[str], List[str], List[str]]:
    r"""Add a set of marks to the screenshot.

    Args:
        screenshot (Image.Image): The screenshot to add marks to.
        ROIs (Dict[str, InteractiveRegion]): The regions to add marks to.

    Returns:
        Tuple[Image.Image, List[str], List[str], List[str]]: A tuple
            containing the screenshot with marked ROIs, ROIs fully within the
            images, ROIs located above the visible area, and ROIs located below
            the visible area.
    """
    visible_rects: List[str] = list()
    rects_above: List[str] = list()  # Scroll up to see
    rects_below: List[str] = list()  # Scroll down to see

    fnt = ImageFont.load_default(14)
    base = screenshot.convert("L").convert("RGBA")
    overlay = Image.new("RGBA", base.size)

    draw = ImageDraw.Draw(overlay)
    for r_key in ROIs:  # Renamed r to r_key
        for rect_item in ROIs[r_key]["rects"]:  # Renamed rect to rect_item
            # Empty rectangles
            if (
                    not rect_item
                    or rect_item["width"] == 0
                    or rect_item["height"] == 0
            ):
                continue

            # TODO: add scroll left and right?
            horizontal_center = (rect_item["right"] + rect_item["left"]) / 2.0
            vertical_center = (rect_item["top"] + rect_item["bottom"]) / 2.0
            is_within_horizon = 0 <= horizontal_center < base.size[0]
            is_above_viewport = vertical_center < 0
            is_below_viewport = vertical_center >= base.size[1]

            if is_within_horizon:
                if is_above_viewport:
                    rects_above.append(r_key)
                elif is_below_viewport:
                    rects_below.append(r_key)
                else:  # Fully visible
                    visible_rects.append(r_key)
                    _draw_roi(draw, int(r_key), fnt, rect_item)

    comp = Image.alpha_composite(base, overlay)
    overlay.close()
    return comp, visible_rects, rects_above, rects_below


def _draw_roi(
        draw: ImageDraw.ImageDraw,
        idx: int,
        font: Union[ImageFont.FreeTypeFont, ImageFont.ImageFont],
        # Made Union explicit
        rect: DOMRectangle,
) -> None:
    r"""Draw a ROI on the image.

    Args:
        draw (ImageDraw.ImageDraw): The draw object.
        idx (int): The index of the ROI.
        font (ImageFont.FreeTypeFont | ImageFont.ImageFont): The font.
        rect (DOMRectangle): The DOM rectangle.
    """
    color = _get_random_color(idx)
    text_color = _get_text_color(color)

    roi = ((rect["left"], rect["top"]), (rect["right"], rect["bottom"]))

    label_location = (rect["right"], rect["top"])
    label_anchor = "rb"

    if label_location[1] <= TOP_NO_LABEL_ZONE:
        label_location = (rect["right"], rect["bottom"])
        label_anchor = "rt"

    draw.rectangle(
        roi, outline=color, fill=(color[0], color[1], color[2], 48), width=2
    )

    bbox = draw.textbbox(
        label_location,
        str(idx),
        font=font,
        anchor=label_anchor,
        align="center",
    )
    bbox = (bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3)
    draw.rectangle(bbox, fill=color)

    draw.text(
        label_location,
        str(idx),
        fill=text_color,
        font=font,
        anchor=label_anchor,
        align="center",
    )


def _get_text_color(
        bg_color: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    r"""Determine the ideal text color (black or white) for contrast.

    Args:
        bg_color: The background color (R, G, B, A).

    Returns:
        A tuple representing black or white color for text.
    """
    luminance = bg_color[0] * 0.3 + bg_color[1] * 0.59 + bg_color[2] * 0.11
    return (0, 0, 0, 255) if luminance > 120 else (255, 255, 255, 255)


def _get_random_color(identifier: int) -> Tuple[int, int, int, int]:
    r"""Generate a consistent random RGBA color based on the identifier.

    Args:
        identifier: The ID used as a seed to ensure color consistency.

    Returns:
        A tuple representing (R, G, B, A) values.
    """
    rnd = random.Random(int(identifier))
    r_val = rnd.randint(0, 255)  # Renamed r to r_val
    g_val = rnd.randint(125, 255)  # Renamed g to g_val
    b_val = rnd.randint(0, 50)  # Renamed b to b_val
    color = [r_val, g_val, b_val]
    # TODO: check why shuffle is needed?
    rnd.shuffle(color)
    color.append(255)
    return cast(Tuple[int, int, int, int], tuple(color))


class BaseBrowser:
    def __init__(
            self,
            headless=True,
            cache_dir: Optional[str] = None,
            channel: Literal["chrome", "msedge", "chromium"] = "chromium",
            cookie_json_path: Optional[str] = None,
    ):
        self.history: List[Any] = []
        self.headless = headless
        self.channel = channel
        self._ensure_browser_installed()
        self.playwright = None
        self.page_history: List[str] = []
        self.cookie_json_path = cookie_json_path

        self.cache_dir = "tmp/" if cache_dir is None else cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        abs_dir_path = os.path.dirname(os.path.abspath(__file__))
        page_script_path = os.path.join(abs_dir_path, "page_script.js")

        try:
            with open(page_script_path, "r", encoding='utf-8') as f:
                self.page_script = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Page script file not found at path: {page_script_path}"
            )
        self.browser = None
        self.context = None
        self.page = None
        self.page_url = None
        self.web_agent_model = None

    async def init(self) -> None:
        """Initialize the browser asynchronously."""
        from playwright.async_api import async_playwright

        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            channel=self.channel
        )

        if self.cookie_json_path and os.path.exists(self.cookie_json_path):
            self.context = await self.browser.new_context(
                accept_downloads=True,
                storage_state=self.cookie_json_path
            )
        else:
            self.context = await self.browser.new_context(accept_downloads=True)

        self.page = await self.context.new_page()

    async def _wait_for_load(self, timeout: int = 20) -> None:
        """Wait for page to load asynchronously."""
        try:
            timeout_ms = timeout * 1000
            await self.page.wait_for_load_state("load", timeout=timeout_ms)
            await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"Error during wait for load: {e}")
            raise

    async def click_blank_area(self) -> None:
        """Click a blank area asynchronously."""
        try:
            await self.page.mouse.click(0, 0)
            await self._wait_for_load()
        except Exception as e:
            logger.error(f"Error during click blank area: {e}")
            raise

    async def visit_page(self, url: str) -> None:
        """Visit a page asynchronously."""
        try:
            await self.page.goto(url)
            await self._wait_for_load()
            self.page_url = url
        except Exception as e:
            logger.error(f"Error during visit page: {e}")
            raise

    async def ask_question_about_video(self, question: str) -> str:
        """Ask a question about video asynchronously."""
        current_url = await self.get_url()

        confirmation_message = (
            f"Do you want to analyze the video on the current "
            f"page({current_url})? This operation may take a long time.(y/n): "
        )
        user_confirmation = input(confirmation_message)

        if user_confirmation.lower() not in ['y', 'yes']:
            return "User cancelled the video analysis."

        model = None
        if hasattr(self, 'web_agent_model') and self.web_agent_model is not None:
            model = self.web_agent_model

        video_analyzer = VideoAnalysisTool(model=model)
        result = await video_analyzer.ask_question_about_video(current_url, question)
        return result

    async def get_screenshot(
            self, save_image: bool = False
    ) -> Tuple[Image.Image, Union[str, None]]:
        """Get screenshot asynchronously."""
        image_data = await self.page.screenshot(timeout=60000)
        image = Image.open(io.BytesIO(image_data))

        file_path = None
        if save_image:
            parsed_url = urllib.parse.urlparse(self.page_url)
            url_name = sanitize_filename(str(parsed_url.path), max_length=241)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(
                self.cache_dir, f"{url_name}_{timestamp}.png"
            )
            with open(file_path, "wb") as f:
                image.save(f, "PNG")

        return image, file_path

    async def capture_full_page_screenshots(
            self, scroll_ratio: float = 0.8
    ) -> List[str]:
        """Capture full page screenshots asynchronously."""
        screenshots = []
        scroll_height_eval = await self.page.evaluate("document.body.scrollHeight")
        scroll_height = cast(float, scroll_height_eval)

        viewport_height = self.page.viewport_size["height"]
        current_scroll_eval = await self.page.evaluate("window.scrollY")
        current_scroll = cast(float, current_scroll_eval)

        max_height = scroll_height - viewport_height
        scroll_step = int(viewport_height * scroll_ratio)
        last_height = 0.0

        while True:
            logger.debug(
                f"Current scroll: {current_scroll}, max_height: "
                f"{max_height}, step: {scroll_step}"
            )

            _, file_path = await self.get_screenshot(save_image=True)
            if file_path is not None:
                screenshots.append(file_path)

            await self.page.evaluate(f"window.scrollBy(0, {scroll_step})")
            await asyncio.sleep(0.5)

            current_scroll_eval = await self.page.evaluate("window.scrollY")
            current_scroll = cast(float, current_scroll_eval)
            if abs(current_scroll - last_height) < viewport_height * 0.1:
                break

            last_height = current_scroll

        return screenshots

    async def get_visual_viewport(self) -> VisualViewport:
        """Get visual viewport asynchronously."""
        try:
            await self.page.evaluate(self.page_script)
        except Exception as e:
            logger.warning(f"Error evaluating page script: {e}")

        visual_viewport_eval = await self.page.evaluate(
            "MultimodalWebSurfer.getVisualViewport();"
        )
        return visual_viewport_from_dict(cast(Dict[str, Any], visual_viewport_eval))

    async def get_interactive_elements(self) -> Dict[str, InteractiveRegion]:
        """Get interactive elements asynchronously."""
        try:
            await self.page.evaluate(self.page_script)
        except Exception as e:
            logger.warning(f"Error evaluating page script: {e}")

        result = cast(
            Dict[str, Dict[str, Any]],
            await self.page.evaluate("MultimodalWebSurfer.getInteractiveRects();"),
        )

        typed_results: Dict[str, InteractiveRegion] = {}
        for k in result:
            typed_results[k] = interactive_region_from_dict(result[k])

        return typed_results

    async def get_som_screenshot(
            self,
            save_image: bool = False,
    ) -> Tuple[Image.Image, Union[str, None]]:
        """Get screenshot with interactive elements marked asynchronously."""
        await self._wait_for_load()
        screenshot, _ = await self.get_screenshot(save_image=False)
        rects = await self.get_interactive_elements()

        file_path = None
        comp, _, _, _ = _add_set_of_mark(screenshot, rects)

        if save_image:
            parsed_url = urllib.parse.urlparse(self.page_url)
            url_name = sanitize_filename(str(parsed_url.path), max_length=241)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(
                self.cache_dir, f"{url_name}_{timestamp}.png"
            )
            with open(file_path, "wb") as f:
                comp.save(f, "PNG")

        return comp, file_path

    async def scroll_up(self) -> None:
        """Scroll up asynchronously."""
        try:
            await self.page.keyboard.press("PageUp")
            await self._wait_for_load()
        except Exception as e:
            logger.error(f"Error during scroll up: {e}")
            raise

    async def scroll_down(self) -> None:
        """Scroll down asynchronously."""
        try:
            await self.page.keyboard.press("PageDown")
            await self._wait_for_load()
        except Exception as e:
            logger.error(f"Error during scroll down: {e}")
            raise

    async def get_url(self) -> str:
        """Get current URL asynchronously."""
        return self.page.url

    async def click_id(self, identifier: Union[str, int]) -> None:
        """Click element by ID asynchronously."""
        if isinstance(identifier, int):
            identifier = str(identifier)
        target = self.page.locator(f"[__elementId='{identifier}']")

        try:
            await target.wait_for(timeout=5000)
        except Exception as e:
            logger.debug(f"Error during click operation: {e}")
            raise ValueError("No such element.") from None

        await target.scroll_into_view_if_needed()

        new_page = None
        try:
            async with self.page.expect_event("popup", timeout=1000) as page_info:
                box = await target.bounding_box()
                if box is None:
                    logger.warning(
                        f"Bounding box not found for element '{identifier}'. "
                        f"Cannot click."
                    )
                    return
                await self.page.mouse.click(
                    box["x"] + box["width"] / 2,
                    box["y"] + box["height"] / 2
                )
                new_page = await page_info.value

                if new_page:
                    self.page_history.append(deepcopy(self.page.url))
                    self.page = new_page

        except Exception as e:
            logger.debug(f"Error during click operation: {e}")
            pass

        await self._wait_for_load()

    async def extract_url_content(self) -> str:
        """Extract page content asynchronously."""
        content = await self.page.content()
        return content

    async def download_file_id(self, identifier: Union[str, int]) -> str:
        """Download file by ID asynchronously."""
        if isinstance(identifier, int):
            identifier = str(identifier)
        try:
            target = self.page.locator(f"[__elementId='{identifier}']")
        except Exception as e:
            logger.debug(f"Error during download operation: {e}")
            logger.warning(f"Element with identifier '{identifier}' not found.")
            return f"Element with identifier '{identifier}' not found."

        await target.scroll_into_view_if_needed()

        file_path_val = os.path.join(self.cache_dir)
        await self._wait_for_load()

        try:
            async with self.page.expect_download() as download_info:
                await target.click()
                download = await download_info.value
                file_name = download.suggested_filename

                file_path_val = os.path.join(file_path_val, file_name)
                await download.save_as(file_path_val)

            return f"Downloaded file to path '{file_path_val}'."

        except Exception as e:
            logger.debug(f"Error during download operation: {e}")
            return f"Failed to download file with identifier '{identifier}'."

    async def fill_input_id(self, identifier: Union[str, int], text: str) -> str:
        """Fill input by ID asynchronously."""
        if isinstance(identifier, int):
            identifier = str(identifier)

        try:
            target = self.page.locator(f"[__elementId='{identifier}']")
        except Exception as e:
            logger.debug(f"Error during fill operation: {e}")
            logger.warning(f"Element with identifier '{identifier}' not found.")
            return f"Element with identifier '{identifier}' not found."

        await target.scroll_into_view_if_needed()
        await target.focus()
        try:
            await target.fill(text)
        except Exception as e:
            logger.debug(f"Error during fill operation: {e}")
            await target.press_sequentially(text)

        await target.press("Enter")
        await self._wait_for_load()
        return (
            f"Filled input field '{identifier}' with text '{text}' "
            f"and pressed Enter."
        )

    async def scroll_to_bottom(self) -> str:
        """Scroll to bottom asynchronously."""
        try:
            await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
            await self._wait_for_load()
            return "Scrolled to the bottom of the page."
        except Exception as e:
            logger.error(f"Error during scroll to bottom: {e}")
            raise

    async def scroll_to_top(self) -> str:
        """Scroll to top asynchronously."""
        try:
            await self.page.evaluate("window.scrollTo(0, 0);")
            await self._wait_for_load()
            return "Scrolled to the top of the page."
        except Exception as e:
            logger.error(f"Error during scroll to top: {e}")
            raise

    async def hover_id(self, identifier: Union[str, int]) -> str:
        """Hover over element by ID asynchronously."""
        if isinstance(identifier, int):
            identifier = str(identifier)
        try:
            target = self.page.locator(f"[__elementId='{identifier}']")
        except Exception as e:
            logger.debug(f"Error during hover operation: {e}")
            logger.warning(f"Element with identifier '{identifier}' not found.")
            return f"Element with identifier '{identifier}' not found."

        await target.scroll_into_view_if_needed()
        await target.hover()
        await self._wait_for_load()
        return f"Hovered over element with identifier '{identifier}'."

    async def find_text_on_page(self, search_text: str) -> str:
        """Find text on page asynchronously."""
        script = f"""
        (function() {{ 
            let text = "{search_text}";
            let found = window.find(text);
            if (!found) {{
                let elements = document.querySelectorAll("*:not(script):not(style)"); 
                for (let el of elements) {{
                    if (el.innerText && el.innerText.includes(text)) {{
                        el.scrollIntoView({{behavior: "smooth", block: "center"}});
                        el.style.backgroundColor = "yellow";
                        el.style.border = '2px solid red';
                        return true;
                    }}
                }}
                return false;
            }}
            return true;
        }})();
        """
        found_eval = await self.page.evaluate(script)
        found = cast(bool, found_eval)
        await self._wait_for_load()
        if found:
            return f"Found text '{search_text}' on the page."
        else:
            return f"Text '{search_text}' not found on the page."

    async def back(self):
        """Navigate back asynchronously."""
        page_url_before = self.page.url
        await self.page.go_back()

        page_url_after = self.page.url

        if page_url_after == "about:blank":
            await self.visit_page(page_url_before)

        if page_url_before == page_url_after:
            if len(self.page_history) > 0:
                await self.visit_page(self.page_history.pop())

        await asyncio.sleep(1)
        await self._wait_for_load()

    async def close(self):
        """Close browser asynchronously."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def show_interactive_elements(self):
        """Show interactive elements asynchronously."""
        await self.page.evaluate(self.page_script)
        await self.page.evaluate("""
        () => {
            document.querySelectorAll('a, button, input, select, textarea, [tabindex]:not([tabindex="-1"]), [contenteditable="true"]').forEach(el => {
                el.style.border = '2px solid red';
            });
        }
        """)

    async def get_webpage_content(self) -> str:
        """Get webpage content asynchronously."""
        from html2text import html2text

        await self._wait_for_load()
        html_content = await self.page.content()

        markdown_content = html2text(html_content)
        return markdown_content

    def _ensure_browser_installed(self) -> None:
        """Ensure the browser is installed."""
        import platform
        import subprocess
        import sys
        from pathlib import Path

        path = os.path.join(AGENTICA_HOME, "browser")
        cache_dir = Path(path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.channel}_installed"

        if cache_file.exists():
            import time
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age < 7 * 24 * 60 * 60:  # 7 days in seconds
                return

        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(channel=self.channel)
                browser.close()
                cache_file.touch()
        except Exception:
            logger.info("Installing Chromium browser...")
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "playwright",
                        "install",
                        self.channel,
                    ],
                    check=True,
                    capture_output=True,
                )
                if platform.system().lower() == "linux":
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "playwright",
                            "install-deps",
                            self.channel,
                        ],
                        check=True,
                        capture_output=True,
                    )
                logger.info("Chromium browser installation completed")
                # 安装成功后创建缓存标记
                cache_file.touch()
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to install browser: {e.stderr}")


class Browser:
    r"""A class for browsing the web and interacting with web pages.

    This class provides methods for browsing the web and interacting with web
    pages.
    """

    def __init__(
            self,
            headless: bool = False,
            cache_dir: Optional[str] = None,
            channel: Literal["chrome", "msedge", "chromium"] = "chromium",
            history_window: int = 5,
            web_agent_model: Optional[Model] = None,
            planning_agent_model: Optional[Model] = None,
            output_language: str = "en",
            cookie_json_path: Optional[str] = None,
    ):
        r"""Initialize the BrowserToolkit instance.

        Args:
            headless (bool): Whether to run the browser in headless mode.
            cache_dir (Union[str, None]): The directory to store cache files.
            channel (Literal["chrome", "msedge", "chromium"]): The browser
                channel to use. Must be one of "chrome", "msedge", or
                "chromium".
            history_window (int): The window size for storing the history of
                actions.
            web_agent_model (Optional[Model]): The model backend
                for the web agent.
            planning_agent_model (Optional[Model]): The model
                backend for the planning agent.
            output_language (str): The language to use for output.
                (default: :obj:`"en`")
            cookie_json_path (Optional[str]): Path to a JSON file containing
                authentication cookies and browser storage state. If provided
                and the file exists, the browser will load this state to
                maintain
                authenticated sessions without requiring manual login.
                (default: :obj:`None`)
        """
        self.browser = BaseBrowser(
            headless=headless,
            cache_dir=cache_dir,
            channel=channel,
            cookie_json_path=cookie_json_path,
        )
        self.browser.web_agent_model = web_agent_model

        self.history_window = history_window
        self.web_agent_model = web_agent_model
        self.planning_agent_model = planning_agent_model
        self.output_language = output_language

        self.history: List[Dict[str, Any]] = []
        self.web_agent: Agent
        self.planning_agent: Agent
        self.web_agent, self.planning_agent = self._initialize_agent(
            web_agent_model, planning_agent_model
        )

    def _reset(self):
        self.web_agent.reset()
        self.planning_agent.reset()
        self.history = []
        os.makedirs(self.browser.cache_dir, exist_ok=True)

    def _initialize_agent(
            self,
            web_agent_model_backend: Optional[Model],
            planning_agent_model_backend: Optional[Model],
    ) -> Tuple[Agent, Agent]:
        if web_agent_model_backend is None:
            web_agent_model_instance = OpenAIChat()
        else:
            web_agent_model_instance = web_agent_model_backend

        if planning_agent_model_backend is None:
            planning_model = OpenAIChat()
        else:
            planning_model = planning_agent_model_backend

        web_agent = Agent(
            system_message=WEB_AGENT_SYSTEM_PROMPT,
            model=web_agent_model_instance,
            output_language=self.output_language,
        )

        planning_agent = Agent(
            system_message=PLANNING_AGENT_SYSTEM_PROMPT,
            model=planning_model,
            output_language=self.output_language,
        )

        return web_agent, planning_agent

    async def _observe(
            self, task_prompt: str, detailed_plan: Optional[str] = None
    ) -> Tuple[str, str, str]:
        """Let agent observe the current environment and get the next action."""
        detailed_plan_prompt_str = ""

        if detailed_plan is not None:
            detailed_plan_prompt_str = f"""
Here is a plan about how to solve the task step-by-step which you must follow:
<detailed_plan>{detailed_plan}<detailed_plan>
            """

        observe_prompt = OBSERVE_PROMPT_TEMPLATE.format(
            task_prompt=task_prompt,
            detailed_plan_prompt=detailed_plan_prompt_str,
            AVAILABLE_ACTIONS_PROMPT=AVAILABLE_ACTIONS_PROMPT,
            history_window=self.history_window,
            history=self.history[-self.history_window:],
        )

        som_screenshot, _ = await self.browser.get_som_screenshot(save_image=True)
        img = _reload_image(som_screenshot)
        message = UserMessage(content=observe_prompt, images=[img])
        self.web_agent.reset()
        resp = await self.web_agent.arun(message)
        resp_content = resp.content

        resp_dict = _parse_json_output(resp_content, logger)
        observation_result: str = resp_dict.get("observation", "")
        reasoning_result: str = resp_dict.get("reasoning", "")
        action_code: str = resp_dict.get("action_code", "")

        if action_code and "(" in action_code and ")" not in action_code:
            action_match = re.search(
                r'"action_code"\s*:\s*[`"]([^`"]*\([^)]*\))[`"]', resp_content
            )
            if action_match:
                action_code = action_match.group(1)
            else:
                logger.warning(f"Incomplete action_code detected: {action_code}")
                if action_code.startswith("fill_input_id("):
                    parts = action_code.split(",", 1)
                    if len(parts) > 1:
                        id_part = parts[0].replace("fill_input_id(", "").strip()
                        action_code = (
                            f"fill_input_id({id_part}, 'Please fill the text here.')"
                        )

        action_code = action_code.replace("`", "").strip()
        return observation_result, reasoning_result, action_code

    async def _act(self, action_code: str) -> Tuple[bool, str]:
        """Execute the action code asynchronously."""

        def _check_if_with_feedback(action_code: str) -> bool:
            for action_with_feedback in ACTION_WITH_FEEDBACK_LIST:
                if action_with_feedback in action_code:
                    return True
            return False

        def _fix_action_code(action_code: str) -> str:
            match = re.match(r'(\w+)\((.*)\)', action_code)
            if not match:
                return action_code

            func_name, args_str = match.groups()
            args = []
            current_arg = ""
            in_quotes = False
            quote_char = None

            for char in args_str:
                if char in ['"', "'"]:
                    if not in_quotes:
                        in_quotes = True
                        quote_char = char
                        current_arg += char
                    elif char == quote_char:
                        in_quotes = False
                        quote_char = None
                        current_arg += char
                    else:
                        current_arg += char
                elif char == ',' and not in_quotes:
                    args.append(current_arg.strip())
                    current_arg = ""
                else:
                    current_arg += char

            if current_arg:
                args.append(current_arg.strip())

            fixed_args = []
            for arg in args:
                if (
                        (arg.startswith('"') and arg.endswith('"'))
                        or (arg.startswith("'") and arg.endswith("'"))
                        or re.match(r'^-?\d+(\.\d+)?$', arg)
                        or re.match(r'^-?\d+\.?\d*[eE][-+]?\d+$', arg)
                        or re.match(r'^0[xX][0-9a-fA-F]+$', arg)
                ):
                    fixed_args.append(arg)
                else:
                    fixed_args.append(f"'{arg}'")

            return f"{func_name}({', '.join(fixed_args)})"

        action_code = _fix_action_code(action_code)
        prefix = "self.browser."
        code = f"{prefix}{action_code}"

        try:
            if _check_if_with_feedback(action_code):
                result = await eval(code)
                return True, result
            else:
                await exec(code)
                return True, "Action was successful."

        except Exception as e:
            return (
                False,
                f"Error while executing the action {action_code}: {e}. "
                f"If timeout, please recheck whether you have provided the "
                f"correct identifier.",
            )

    async def _get_final_answer(self, task_prompt: str) -> str:
        """Get the final answer based on the task prompt and current browser state."""
        prompt = GET_FINAL_ANSWER_PROMPT_TEMPLATE.format(
            history=self.history, task_prompt=task_prompt
        )
        message = UserMessage(content=prompt)
        self.web_agent.reset()
        resp = await self.web_agent.arun(message)
        return resp.content

    async def _task_planning(self, task_prompt: str, start_url: str) -> str:
        """Plan the task based on the given task prompt."""
        planning_prompt = TASK_PLANNING_PROMPT_TEMPLATE.format(
            task_prompt=task_prompt, start_url=start_url
        )

        message = UserMessage(content=planning_prompt)
        self.planning_agent.reset()
        resp = await self.planning_agent.arun(message)
        return resp.content

    async def _task_replanning(
            self, task_prompt: str, detailed_plan: str
    ) -> Tuple[bool, str]:
        """Replan the task based on the given task prompt."""
        replanning_prompt = TASK_REPLANNING_PROMPT_TEMPLATE.format(
            task_prompt=task_prompt,
            detailed_plan=detailed_plan,
            history_window=self.history_window,
            history=self.history[-self.history_window:],
        )
        self.planning_agent.reset()
        resp = await self.planning_agent.arun(replanning_prompt)
        resp_dict = _parse_json_output(resp.content, logger)

        if_need_replan_eval = resp_dict.get("if_need_replan", False)
        if_need_replan = cast(bool, if_need_replan_eval)
        replanned_schema: str = resp_dict.get("replanned_schema", "")

        if if_need_replan:
            return True, replanned_schema
        else:
            return False, replanned_schema

    async def browse_url(
            self, task_prompt: str, start_url: str, round_limit: int = 12
    ) -> str:
        """A powerful toolkit which can simulate the browser interaction."""
        self._reset()
        task_completed = False
        detailed_plan = await self._task_planning(task_prompt, start_url)
        logger.debug(f"Detailed plan: {detailed_plan}")

        await self.browser.init()
        await self.browser.visit_page(start_url)

        for i in range(round_limit):
            observation, reasoning, action_code = await self._observe(
                task_prompt, detailed_plan
            )
            logger.debug(f"Observation: {observation}")
            logger.debug(f"Reasoning: {reasoning}")
            logger.debug(f"Action code: {action_code}")

            if "stop" in action_code:
                task_completed = True
                trajectory_info = {
                    "round": i,
                    "observation": observation,
                    "thought": reasoning,
                    "action": action_code,
                    "action_if_success": True,
                    "info": None,
                    "current_url": await self.browser.get_url(),
                }
                self.history.append(trajectory_info)
                break

            else:
                success, info = await self._act(action_code)
                if not success:
                    logger.warning(f"Error while executing the action: {info}")

                trajectory_info = {
                    "round": i,
                    "observation": observation,
                    "thought": reasoning,
                    "action": action_code,
                    "action_if_success": success,
                    "info": info,
                    "current_url": await self.browser.get_url(),
                }
                self.history.append(trajectory_info)

                if_need_replan, replanned_schema = await self._task_replanning(
                    task_prompt, detailed_plan
                )
                if if_need_replan:
                    detailed_plan = replanned_schema
                    logger.debug(f"Replanned schema: {replanned_schema}")

        if not task_completed:
            simulation_result = f"""
                The task is not completed within the round limit. Please 
                check the last round {self.history_window} information to 
                see if there is any useful information:
                <history>{self.history[-self.history_window:]}</history>
            """
        else:
            simulation_result = await self._get_final_answer(task_prompt)

        await self.browser.close()
        return simulation_result


class BrowserTool(Tool):
    r"""A class for browsing the web and interacting with web pages.

    This class provides methods for browsing the web and interacting with web
    pages.
    """

    def __init__(
            self,
            headless: bool = False,
            cache_dir: Optional[str] = None,
            channel: Literal["chrome", "msedge", "chromium"] = "chromium",
            history_window: int = 5,
            web_agent_model: Optional[Model] = None,
            planning_agent_model: Optional[Model] = None,
            output_language: str = "en",
            cookie_json_path: Optional[str] = None,
    ):
        r"""Initialize the BrowserToolkit instance.

        Args:
            headless (bool): Whether to run the browser in headless mode.
            cache_dir (Union[str, None]): The directory to store cache files.
            channel (Literal["chrome", "msedge", "chromium"]): The browser
                channel to use. Must be one of "chrome", "msedge", or
                "chromium".
            history_window (int): The window size for storing the history of
                actions.
            web_agent_model (Optional[Model]): The model backend
                for the web agent.
            planning_agent_model (Optional[Model]): The model
                backend for the planning agent.
            output_language (str): The language to use for output.
                (default: :obj:`en`)
            cookie_json_path (Optional[str]): Path to a JSON file containing
                authentication cookies and browser storage state. If provided
                and the file exists, the browser will load this state to
                maintain
                authenticated sessions without requiring manual login.
                (default: :obj:`None`)
        """
        super().__init__(name="BrowserTool")
        self._browser = None
        self.headless = headless
        self.cache_dir = cache_dir
        self.channel = channel
        self.history_window = history_window
        self.web_agent_model = web_agent_model
        self.planning_agent_model = planning_agent_model
        self.output_language = output_language
        self.cookie_json_path = cookie_json_path
        self._event_loop = None
        self.register(self.browse_webpage)

    async def _init_browser(self):
        """Initialize the browser instance."""
        if self._browser is None:
            self._browser = Browser(
                headless=self.headless,
                cache_dir=self.cache_dir,
                channel=self.channel,
                history_window=self.history_window,
                web_agent_model=self.web_agent_model,
                planning_agent_model=self.planning_agent_model,
                output_language=self.output_language,
                cookie_json_path=self.cookie_json_path,
            )
            await self._browser.browser.init()

    def browse_webpage(self, task: str, url: str, round_limit: int = 8) -> str:
        """
        Browse a webpage and perform actions based on the task.
        Args:
            task (str): The task to perform on the webpage.
            url (str): The starting URL for browsing.
            round_limit (int): The maximum number of rounds to perform
                actions. Default is 8.
        Returns:
            str: The result of the browsing task.
        """

        async def _async_browse():
            try:
                await self._init_browser()
                result = await self._browser.browse_url(
                    task_prompt=task,
                    start_url=url,
                    round_limit=round_limit
                )
                return result
            except Exception as e:
                logger.error(f"Error during browser task: {e}")
                raise
            finally:
                if self._browser and self._browser.browser:
                    await self._browser.browser.close()
                    self._browser = None

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(_async_browse())
        except Exception as e:
            logger.error(f"Error in browse_webpage: {e}")
            raise
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))


if __name__ == '__main__':
    m = BrowserTool(
        web_agent_model=OpenAIChat(),
        planning_agent_model=OpenAIChat(),
        cookie_json_path="tmp/cookie.json",
    )
    result = m.browse_webpage(
        task="访问中国天气网(http://www.weather.com.cn)，查看北京今天的天气情况，包括温度、空气质量等信息",
        url="https://www.weather.com.cn",
    )
    print(f"Task result: {result}")
