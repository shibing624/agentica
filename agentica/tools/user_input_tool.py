# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: User Input Tool - Human-in-the-loop tool for agent interactions

This tool enables agents to request input or confirmation from users during execution.
It supports various interaction modes:
- Confirmation: Yes/No questions for critical operations
- Text Input: Free-form text input from users
- Selection: Choose from predefined options

Example:
    ```python
    from agentica import DeepAgent
    from agentica.tools.user_input_tool import UserInputTool
    
    agent = DeepAgent(
        tools=[UserInputTool()],
        instructions="When uncertain, ask the user for confirmation.",
    )
    ```
"""
import json
from typing import Optional, List, Callable, Literal

from agentica.tools.base import Tool, StopAgentRun
from agentica.model.message import Message
from agentica.utils.log import logger


class UserInputTool(Tool):
    """
    Human-in-the-loop tool that allows agents to request user input during execution.
    
    This tool provides several interaction modes:
    1. **Confirmation**: Ask yes/no questions for critical decisions
    2. **Text Input**: Request free-form text from users
    3. **Selection**: Present options for users to choose from
    
    The tool uses a callback mechanism to get user input. If no callback is provided,
    it defaults to console input (useful for CLI applications).
    
    Attributes:
        input_callback: Custom callback function for getting user input.
                       Signature: (prompt: str, options: Optional[List[str]]) -> str
        timeout: Timeout in seconds for waiting for user input (default: 300)
        default_on_timeout: Default value to return if timeout occurs
    
    Example:
        ```python
        # Basic usage with console input
        tool = UserInputTool()
        
        # With custom callback (e.g., for web applications)
        def web_input_callback(prompt: str, options: Optional[List[str]] = None) -> str:
            # Send prompt to frontend and wait for response
            return frontend_api.get_user_input(prompt, options)
        
        tool = UserInputTool(input_callback=web_input_callback)
        ```
    """
    
    # System prompt for user input tool usage guidance
    USER_INPUT_SYSTEM_PROMPT = """## `user_input` (Human-in-the-loop)

You have access to the `user_input` tool to request input or confirmation from the user during execution.

### When to use this tool:
1. **Critical Operations**: Before performing irreversible actions (delete files, send emails, make purchases)
2. **Ambiguous Requests**: When the user's intent is unclear and you need clarification
3. **Sensitive Information**: When you need passwords, API keys, or personal information
4. **Decision Points**: When multiple valid approaches exist and user preference matters
5. **Verification**: To confirm understanding of complex requirements before proceeding

### Interaction Modes:
- `confirm`: Yes/No questions - use for binary decisions
- `text`: Free-form input - use when you need detailed information
- `select`: Multiple choice - use when there are specific options to choose from

### Best Practices:
- Provide clear, concise prompts that explain what you need and why
- For confirmations, clearly state what action will be taken
- For selections, provide meaningful option descriptions
- Don't overuse - only ask when truly necessary
- Group related questions when possible to minimize interruptions

### Example Usage:
```
# Confirmation before deletion
user_input(prompt="Delete all files in /tmp/old_data? This cannot be undone.", mode="confirm")

# Get specific information
user_input(prompt="Please provide your preferred output format:", mode="select", options=["JSON", "CSV", "XML"])

# Free-form input
user_input(prompt="Describe the specific changes you want to make to the code:", mode="text")
```"""

    def __init__(
        self,
        input_callback: Optional[Callable[[str, Optional[List[str]]], str]] = None,
        timeout: int = 300,
        default_on_timeout: Optional[str] = None,
    ):
        """
        Initialize UserInputTool.
        
        Args:
            input_callback: Custom callback function for getting user input.
                           If None, uses console input.
            timeout: Timeout in seconds for waiting for user input.
            default_on_timeout: Default value to return if timeout occurs.
        """
        super().__init__(name="user_input_tool")
        self.input_callback = input_callback
        self.timeout = timeout
        self.default_on_timeout = default_on_timeout
        
        # Register the user_input function
        self.register(self.user_input)
        self.register(self.confirm)
    
    def get_system_prompt(self) -> Optional[str]:
        """Get the system prompt for user input tool usage guidance."""
        return self.USER_INPUT_SYSTEM_PROMPT
    
    def _get_input(self, prompt: str, options: Optional[List[str]] = None) -> str:
        """
        Internal method to get user input using callback or console.
        
        Args:
            prompt: The prompt to display to the user
            options: Optional list of valid options
            
        Returns:
            User's input as a string
        """
        if self.input_callback is not None:
            try:
                return self.input_callback(prompt, options)
            except Exception as e:
                logger.error(f"Error in input callback: {e}")
                if self.default_on_timeout:
                    return self.default_on_timeout
                raise
        
        # Default console input
        print("\n" + "=" * 60)
        print("🤖 Agent is requesting your input:")
        print("-" * 60)
        print(prompt)
        
        if options:
            print("\nAvailable options:")
            for i, opt in enumerate(options, 1):
                print(f"  {i}. {opt}")
            print(f"\nEnter option number (1-{len(options)}) or type your choice:")
        
        print("-" * 60)
        
        try:
            user_input = input("Your response: ").strip()
            
            # Handle option number input
            if options and user_input.isdigit():
                idx = int(user_input) - 1
                if 0 <= idx < len(options):
                    user_input = options[idx]
            
            print("=" * 60 + "\n")
            return user_input
        except EOFError:
            # Handle non-interactive environments
            logger.warning("Non-interactive environment detected, using default")
            if self.default_on_timeout:
                return self.default_on_timeout
            return ""
    
    def user_input(
        self,
        prompt: str,
        mode: Literal["confirm", "text", "select"] = "text",
        options: Optional[List[str]] = None,
    ) -> str:
        """
        Request input from the user during agent execution.
        
        This function pauses agent execution and waits for user input.
        Use this when you need clarification, confirmation, or additional
        information from the user to proceed with a task.
        
        Args:
            prompt: Clear description of what input is needed and why.
                   For confirmations, describe the action that will be taken.
            mode: Type of input requested:
                - "confirm": Yes/No question (returns "yes" or "no")
                - "text": Free-form text input
                - "select": Choose from options (requires options parameter)
            options: List of valid options for "select" mode.
                    Each option should be a clear, concise description.
        
        Returns:
            str: User's response:
                - For "confirm": "yes" or "no"
                - For "text": The user's input text
                - For "select": The selected option
        
        Examples:
            # Confirmation
            response = user_input(
                prompt="Delete all temporary files? This action cannot be undone.",
                mode="confirm"
            )
            
            # Text input
            response = user_input(
                prompt="Please provide the API endpoint URL:",
                mode="text"
            )
            
            # Selection
            response = user_input(
                prompt="Choose the output format:",
                mode="select",
                options=["JSON", "CSV", "XML", "Plain Text"]
            )
        """
        logger.info(f"User input requested: mode={mode}, prompt={prompt[:100]}...")
        
        # Validate mode and options
        if mode == "select" and not options:
            return json.dumps({
                "error": "Options required for select mode",
                "prompt": prompt,
            }, ensure_ascii=False)
        
        # Build the full prompt based on mode
        if mode == "confirm":
            full_prompt = f"{prompt}\n\nPlease respond with 'yes' or 'no'."
            valid_options = ["yes", "no", "y", "n"]
        elif mode == "select":
            full_prompt = prompt
            valid_options = options
        else:  # text mode
            full_prompt = prompt
            valid_options = None
        
        # Get user input
        response = self._get_input(full_prompt, valid_options if mode == "select" else None)
        
        # Normalize confirmation responses
        if mode == "confirm":
            response_lower = response.lower().strip()
            if response_lower in ["yes", "y", "是", "确认", "ok", "确定"]:
                response = "yes"
            elif response_lower in ["no", "n", "否", "取消", "cancel"]:
                response = "no"
            else:
                # Invalid response, ask again or default to no
                logger.warning(f"Invalid confirmation response: {response}, defaulting to 'no'")
                response = "no"
        
        # Validate selection
        if mode == "select" and options and response not in options:
            # Try to find a close match
            response_lower = response.lower()
            for opt in options:
                if opt.lower() == response_lower or opt.lower().startswith(response_lower):
                    response = opt
                    break
            else:
                logger.warning(f"Invalid selection: {response}, using first option")
                response = options[0] if options else response
        
        logger.info(f"User input received: {response[:100]}...")
        
        return json.dumps({
            "mode": mode,
            "prompt": prompt[:200],
            "response": response,
        }, ensure_ascii=False)
    
    def confirm(self, prompt: str) -> str:
        """
        Quick confirmation method - shorthand for user_input with mode="confirm".
        
        Use this for simple yes/no questions before critical operations.
        
        Args:
            prompt: Description of the action to confirm.
                   Should clearly state what will happen if confirmed.
        
        Returns:
            str: JSON with "response" field containing "yes" or "no"
        
        Example:
            result = confirm("Proceed with deploying to production?")
            # Returns: {"mode": "confirm", "prompt": "...", "response": "yes"}
        """
        return self.user_input(prompt=prompt, mode="confirm")


class UserInputRequired(StopAgentRun):
    """
    Exception raised when user input is required but not available.
    
    This can be used to signal that the agent should pause and wait
    for user input before continuing.
    """
    
    def __init__(
        self,
        prompt: str,
        mode: str = "text",
        options: Optional[List[str]] = None,
    ):
        """
        Initialize UserInputRequired exception.
        
        Args:
            prompt: The prompt to show to the user
            mode: Type of input needed
            options: Options for select mode
        """
        self.prompt = prompt
        self.mode = mode
        self.options = options
        
        user_message = Message(
            role="assistant",
            content=f"I need your input to proceed:\n\n{prompt}"
        )
        
        super().__init__(
            exc=f"User input required: {prompt}",
            user_message=user_message,
        )


if __name__ == "__main__":
    # Test the UserInputTool
    tool = UserInputTool()
    
    print("Testing UserInputTool...")
    
    # Test confirmation
    print("\n--- Test 1: Confirmation ---")
    result = tool.confirm("Do you want to proceed with the test?")
    print(f"Result: {result}")
    
    # Test text input
    print("\n--- Test 2: Text Input ---")
    result = tool.user_input(
        prompt="Please enter your name:",
        mode="text"
    )
    print(f"Result: {result}")
    
    # Test selection
    print("\n--- Test 3: Selection ---")
    result = tool.user_input(
        prompt="Choose your preferred programming language:",
        mode="select",
        options=["Python", "JavaScript", "Go", "Rust"]
    )
    print(f"Result: {result}")
