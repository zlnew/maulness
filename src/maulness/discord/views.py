import asyncio
from typing import Optional
import discord

from maulness.config import config


class ApprovalView(discord.ui.View):
    """Interactive Discord UI component with Approve/Deny buttons for HITL approval."""

    def __init__(self, future: asyncio.Future, timeout: float = 600.0):
        super().__init__(timeout=timeout)
        self.future = future
        self.decision: Optional[bool] = None

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Enforce strict single-user authorization."""
        if config.owner_discord_id and interaction.user.id != config.owner_discord_id:
            await interaction.response.send_message(
                "⛔ Unauthorized: Only Maul can approve actions.", ephemeral=True
            )
            return False
        return True

    @discord.ui.button(label="Approve", style=discord.ButtonStyle.green, emoji="✅")
    async def approve_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.decision = True
        self._disable_all_buttons()
        await interaction.response.edit_message(
            content=f"{interaction.message.content}\n\n**Decision:** [green]Approved by Maul[/green] ✓",
            view=self,
        )
        if not self.future.done():
            self.future.set_result(True)

    @discord.ui.button(label="Deny", style=discord.ButtonStyle.red, emoji="❌")
    async def deny_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.decision = False
        self._disable_all_buttons()
        await interaction.response.edit_message(
            content=f"{interaction.message.content}\n\n**Decision:** [red]Denied by Maul[/red] ✗",
            view=self,
        )
        if not self.future.done():
            self.future.set_result(False)

    async def on_timeout(self):
        self._disable_all_buttons()
        if not self.future.done():
            self.future.set_result(False)

    def _disable_all_buttons(self):
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True
