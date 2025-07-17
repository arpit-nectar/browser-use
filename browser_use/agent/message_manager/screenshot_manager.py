"""Screenshot manager for saving browser screenshots during agent conversations."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import anyio
from browser_use.agent.views import AgentOutput
from browser_use.browser.views import BrowserStateSummary
from browser_use.llm.messages import BaseMessage
from browser_use.observability import observe_debug

logger = logging.getLogger(__name__)


class ScreenshotManager:
    """Manages saving screenshots alongside conversation logs."""

    def __init__(self, conversation_dir: Path, agent_id: str):
        """Initialize the screenshot manager.

        Args:
            conversation_dir: Directory where conversation logs are saved
            agent_id: Unique identifier for the agent session
        """
        self.conversation_dir = Path(conversation_dir)
        self.agent_id = agent_id
        self.screenshots_dir = self.conversation_dir / "screenshots"

    @observe_debug(name="save_step_screenshot", ignore_output=True)
    async def save_step_screenshot(
        self,
        browser_state_summary: BrowserStateSummary,
        step_number: int,
        model_output: AgentOutput | None = None,
    ) -> str | None:
        """Save a screenshot for a specific step.

        Args:
            browser_state_summary: Browser state containing screenshot data
            step_number: Current step number
            model_output: Optional model output for additional context

        Returns:
            Path to the saved screenshot file, or None if no screenshot available
        """
        if not browser_state_summary.screenshot:
            logger.debug(f"No screenshot available for step {step_number}")
            return None

        try:
            # Ensure screenshots directory exists
            await anyio.Path(self.screenshots_dir).mkdir(parents=True, exist_ok=True)

            # Create filename with step number and agent ID
            screenshot_filename = f"screenshot_{self.agent_id}_{step_number}.png"
            screenshot_path = self.screenshots_dir / screenshot_filename

            # Decode base64 screenshot and save as PNG
            screenshot_data = base64.b64decode(browser_state_summary.screenshot)
            await anyio.Path(screenshot_path).write_bytes(screenshot_data)

            logger.debug(f"Screenshot saved: {screenshot_path}")
            return str(screenshot_path)

        except Exception as e:
            logger.warning(f"Failed to save screenshot for step {step_number}: {e}")
            return None

    @observe_debug(name="save_conversation_with_screenshots", ignore_output=True)
    async def save_conversation_with_screenshots(
        self,
        input_messages: list[BaseMessage],
        response: Any,
        browser_state_summary: BrowserStateSummary,
        step_number: int,
        encoding: str | None = None,
    ) -> tuple[str, str | None]:
        """Save both conversation text and screenshot for a step.

        Args:
            input_messages: List of input messages
            response: Agent response
            browser_state_summary: Browser state with screenshot
            step_number: Current step number
            encoding: Text encoding to use

        Returns:
            Tuple of (conversation_file_path, screenshot_file_path)
        """
        # Save conversation text
        conversation_filename = f"conversation_{self.agent_id}_{step_number}.txt"
        conversation_path = self.conversation_dir / conversation_filename

        # Create folders if not exists
        await anyio.Path(self.conversation_dir).mkdir(parents=True, exist_ok=True)

        # Format and save conversation
        conversation_text = await self._format_conversation_with_screenshot_refs(input_messages, response, step_number)
        await anyio.Path(conversation_path).write_text(
            conversation_text,
            encoding=encoding or "utf-8",
        )

        # Save screenshot
        screenshot_path = await self.save_step_screenshot(browser_state_summary, step_number)

        return str(conversation_path), screenshot_path

    async def _format_conversation_with_screenshot_refs(
        self, messages: list[BaseMessage], response: Any, step_number: int
    ) -> str:
        """Format conversation with references to saved screenshots."""
        lines = []

        # Add screenshot reference at the top
        screenshot_ref = f"screenshots/screenshot_{self.agent_id}_{step_number}.png"
        lines.append(f"SCREENSHOT: {screenshot_ref}")
        lines.append("")

        # Format messages
        for message in messages:
            lines.append(f" {message.role} ")
            lines.append(message.text)
            lines.append("")  # Empty line after each message

        # Format response
        lines.append(" RESPONSE")
        import json

        lines.append(json.dumps(json.loads(response.model_dump_json(exclude_unset=True)), indent=2))

        return "\n".join(lines)

    @observe_debug(name="create_screenshot_index", ignore_output=True)
    async def create_screenshot_index(self) -> None:
        """Create an index file listing all screenshots for the conversation."""
        try:
            if not await anyio.Path(self.screenshots_dir).exists():
                return

            # Get all screenshot files
            screenshot_files = []
            async for path in anyio.Path(self.screenshots_dir).iterdir():
                if path.suffix == ".png" and path.name.startswith(f"screenshot_{self.agent_id}_"):
                    screenshot_files.append(path.name)

            if not screenshot_files:
                return

            # Sort by step number
            screenshot_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

            # Create index file
            index_content = ["# Screenshot Index", ""]
            index_content.append(f"Agent ID: {self.agent_id}")
            index_content.append(f"Total Screenshots: {len(screenshot_files)}")
            index_content.append("")

            for screenshot_file in screenshot_files:
                step_num = screenshot_file.split("_")[-1].split(".")[0]
                index_content.append(f"Step {step_num}: {screenshot_file}")

            index_path = self.screenshots_dir / "index.md"
            await anyio.Path(index_path).write_text("\n".join(index_content))

            logger.info(f"Screenshot index created: {index_path}")

        except Exception as e:
            logger.warning(f"Failed to create screenshot index: {e}")


@observe_debug(name="save_conversation_with_screenshots", ignore_output=True)
async def save_conversation_with_screenshots(
    input_messages: list[BaseMessage],
    response: Any,
    browser_state_summary: BrowserStateSummary,
    target_dir: str | Path,
    agent_id: str,
    step_number: int,
    encoding: str | None = None,
) -> tuple[str, str | None]:
    """Convenience function to save conversation and screenshots.

    Args:
        input_messages: List of input messages
        response: Agent response
        browser_state_summary: Browser state with screenshot
        target_dir: Directory to save files
        agent_id: Agent identifier
        step_number: Current step number
        encoding: Text encoding

    Returns:
        Tuple of (conversation_file_path, screenshot_file_path)
    """
    screenshot_manager = ScreenshotManager(Path(target_dir), agent_id)
    return await screenshot_manager.save_conversation_with_screenshots(
        input_messages, response, browser_state_summary, step_number, encoding
    )
