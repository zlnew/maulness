import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from maulness.discord.views import ApprovalView, SuspendedAfkView


@pytest.mark.asyncio
async def test_approval_view_interaction_check_unauthorized():
    fut = asyncio.Future()
    view = ApprovalView(future=fut)

    mock_interaction = MagicMock()
    mock_interaction.user.id = 99999
    mock_interaction.response.send_message = AsyncMock()

    with patch("maulness.discord.views.config") as mock_cfg:
        mock_cfg.owner_discord_id = 12345
        allowed = await view.interaction_check(mock_interaction)
        assert allowed is False
        mock_interaction.response.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_approval_view_interaction_check_authorized():
    fut = asyncio.Future()
    view = ApprovalView(future=fut)

    mock_interaction = MagicMock()
    mock_interaction.user.id = 12345

    with patch("maulness.discord.views.config") as mock_cfg:
        mock_cfg.owner_discord_id = 12345
        allowed = await view.interaction_check(mock_interaction)
        assert allowed is True


@pytest.mark.asyncio
async def test_approval_view_approve_with_embed():
    fut = asyncio.Future()
    view = ApprovalView(future=fut)

    mock_interaction = MagicMock()
    mock_interaction.user.display_name = "Maul"
    mock_interaction.response.edit_message = AsyncMock()

    embed = discord.Embed(title="Pending Action", description="Details")
    mock_interaction.message.embeds = [embed]

    await view.approve_button.callback(mock_interaction)

    assert view.decision is True
    assert fut.done() and fut.result() is True
    mock_interaction.response.edit_message.assert_awaited_once()
    assert embed.title == "Action Approved (HITL)"


@pytest.mark.asyncio
async def test_approval_view_approve_without_embed():
    fut = asyncio.Future()
    view = ApprovalView(future=fut)

    mock_interaction = MagicMock()
    mock_interaction.user.display_name = "Maul"
    mock_interaction.response.edit_message = AsyncMock()
    mock_interaction.message.embeds = []
    mock_interaction.message.content = (
        "Please approve: rm -rf [green]Approved by Maul[/green]"
    )

    await view.approve_button.callback(mock_interaction)

    assert view.decision is True
    assert fut.done() and fut.result() is True
    mock_interaction.response.edit_message.assert_awaited_once()
    content_arg = mock_interaction.response.edit_message.call_args.kwargs["content"]
    assert "Approved by **Maul**" in content_arg


@pytest.mark.asyncio
async def test_approval_view_deny_with_embed():
    fut = asyncio.Future()
    view = ApprovalView(future=fut)

    mock_interaction = MagicMock()
    mock_interaction.user.display_name = "Maul"
    mock_interaction.response.edit_message = AsyncMock()

    embed = discord.Embed(title="Pending Action", description="Details")
    mock_interaction.message.embeds = [embed]

    await view.deny_button.callback(mock_interaction)

    assert view.decision is False
    assert fut.done() and fut.result() is False
    mock_interaction.response.edit_message.assert_awaited_once()
    assert embed.title == "Action Rejected (HITL)"


@pytest.mark.asyncio
async def test_approval_view_deny_without_embed():
    fut = asyncio.Future()
    view = ApprovalView(future=fut)

    mock_interaction = MagicMock()
    mock_interaction.user.display_name = "Maul"
    mock_interaction.response.edit_message = AsyncMock()
    mock_interaction.message.embeds = []
    mock_interaction.message.content = (
        "Please approve: drop database [red]Denied by Maul[/red]"
    )

    await view.deny_button.callback(mock_interaction)

    assert view.decision is False
    assert fut.done() and fut.result() is False
    mock_interaction.response.edit_message.assert_awaited_once()
    content_arg = mock_interaction.response.edit_message.call_args.kwargs["content"]
    assert "Rejected by **Maul**" in content_arg


@pytest.mark.asyncio
async def test_approval_view_on_timeout():
    fut = asyncio.Future()
    view = ApprovalView(future=fut)

    await view.on_timeout()

    assert fut.done() and fut.result() is False
    for child in view.children:
        if isinstance(child, discord.ui.Button):
            assert child.disabled is True


@pytest.mark.asyncio
async def test_suspended_afk_view_buttons():
    fut = asyncio.Future()
    view = SuspendedAfkView(task_id="task-afk-99", future=fut)

    mock_interaction = MagicMock()
    mock_interaction.user.display_name = "Maul"
    mock_interaction.response.edit_message = AsyncMock()

    embed = discord.Embed(title="Task Suspended", description="Details")
    mock_interaction.message.embeds = [embed]

    # Test resume
    await view.resume_button.callback(mock_interaction)
    assert view.action == "resume"
    assert fut.done() and fut.result() == "resume"
    mock_interaction.response.edit_message.assert_awaited_once()

    # Test merge on fresh view
    fut2 = asyncio.Future()
    view2 = SuspendedAfkView(task_id="task-afk-99", future=fut2)
    await view2.merge_button.callback(mock_interaction)
    assert view2.action == "merge"
    assert fut2.done() and fut2.result() == "merge"

    # Test abort on fresh view
    fut3 = asyncio.Future()
    view3 = SuspendedAfkView(task_id="task-afk-99", future=fut3)
    await view3.abort_button.callback(mock_interaction)
    assert view3.action == "abort"
    assert fut3.done() and fut3.result() == "abort"
