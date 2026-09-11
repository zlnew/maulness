import asyncio
import datetime
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
        approver = interaction.user.display_name
        if interaction.message.embeds:
            embed = interaction.message.embeds[0]
            embed.color = 0x10B981  # Emerald Green
            embed.title = "✅ Action Approved (HITL)"
            embed.add_field(name="Decision", value=f"Approved by **{approver}**", inline=False)
            embed.timestamp = datetime.datetime.now()
            await interaction.response.edit_message(embed=embed, view=self)
        else:
            clean_content = interaction.message.content.replace("[green]Approved by Maul[/green] ✓", "").strip()
            await interaction.response.edit_message(
                content=clean_content + "\n\n**Decision:** ✅ Approved by **" + approver + "**",
                view=self,
            )
        if not self.future.done():
            self.future.set_result(True)

    @discord.ui.button(label="Deny", style=discord.ButtonStyle.red, emoji="❌")
    async def deny_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.decision = False
        self._disable_all_buttons()
        denier = interaction.user.display_name
        if interaction.message.embeds:
            embed = interaction.message.embeds[0]
            embed.color = 0xEF4444  # Red
            embed.title = "❌ Action Rejected (HITL)"
            embed.add_field(name="Decision", value=f"Rejected by **{denier}**", inline=False)
            embed.timestamp = datetime.datetime.now()
            await interaction.response.edit_message(embed=embed, view=self)
        else:
            clean_content = interaction.message.content.replace("[red]Denied by Maul[/red] ✗", "").strip()
            await interaction.response.edit_message(
                content=clean_content + "\n\n**Decision:** ❌ Rejected by **" + denier + "**",
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
