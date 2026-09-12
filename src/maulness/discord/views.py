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
                "Unauthorized: Only Maul can approve actions.", ephemeral=True
            )
            return False
        return True

    @discord.ui.button(label="Approve", style=discord.ButtonStyle.green)
    async def approve_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        self.decision = True
        self._disable_all_buttons()
        approver = interaction.user.display_name
        if interaction.message.embeds:
            embed = interaction.message.embeds[0]
            embed.color = 0x10B981  # Emerald Green
            embed.title = "Action Approved (HITL)"
            embed.add_field(
                name="Decision", value=f"Approved by **{approver}**", inline=False
            )
            embed.timestamp = datetime.datetime.now()
            await interaction.response.edit_message(embed=embed, view=self)
        else:
            clean_content = interaction.message.content.replace(
                "[green]Approved by Maul[/green]", ""
            ).strip()
            await interaction.response.edit_message(
                content=f"{clean_content}\n\n**Decision:** Approved by **{approver}**",
                view=self,
            )
        if not self.future.done():
            self.future.set_result(True)

    @discord.ui.button(label="Deny", style=discord.ButtonStyle.red)
    async def deny_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        self.decision = False
        self._disable_all_buttons()
        denier = interaction.user.display_name
        if interaction.message.embeds:
            embed = interaction.message.embeds[0]
            embed.color = 0xEF4444  # Red
            embed.title = "Action Rejected (HITL)"
            embed.add_field(
                name="Decision", value=f"Rejected by **{denier}**", inline=False
            )
            embed.timestamp = datetime.datetime.now()
            await interaction.response.edit_message(embed=embed, view=self)
        else:
            clean_content = interaction.message.content.replace(
                "[red]Denied by Maul[/red]", ""
            ).strip()
            await interaction.response.edit_message(
                content=f"{clean_content}\n\n**Decision:** Rejected by **{denier}**",
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


class SuspendedAfkView(discord.ui.View):
    """Interactive Discord UI component for tasks parked in SUSPENDED_AFK status."""

    def __init__(
        self,
        task_id: str,
        future: Optional[asyncio.Future] = None,
        timeout: float = 3600.0,
    ):
        super().__init__(timeout=timeout)
        self.task_id = task_id
        self.future = future
        self.action: Optional[str] = None

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if config.owner_discord_id and interaction.user.id != config.owner_discord_id:
            await interaction.response.send_message(
                "Unauthorized: Only Maul can manage suspended tasks.", ephemeral=True
            )
            return False
        return True

    @discord.ui.button(
        label="Resume Task", style=discord.ButtonStyle.green, custom_id="afk_resume"
    )
    async def resume_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        self.action = "resume"
        self._disable_all_buttons()
        if interaction.message.embeds:
            embed = interaction.message.embeds[0]
            embed.color = 0x10B981
            embed.title = f"Task Resumed: {self.task_id}"
            embed.add_field(
                name="Action",
                value=f"Resumed by **{interaction.user.display_name}**",
                inline=False,
            )
            await interaction.response.edit_message(embed=embed, view=self)
        if self.future and not self.future.done():
            self.future.set_result("resume")

    @discord.ui.button(
        label="Merge / Sign Off",
        style=discord.ButtonStyle.blurple,
        custom_id="afk_merge",
    )
    async def merge_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        self.action = "merge"
        self._disable_all_buttons()
        if interaction.message.embeds:
            embed = interaction.message.embeds[0]
            embed.color = 0x3B82F6
            embed.title = f"Task Approved / Merged: {self.task_id}"
            embed.add_field(
                name="Action",
                value=f"Signed off by **{interaction.user.display_name}**",
                inline=False,
            )
            await interaction.response.edit_message(embed=embed, view=self)
        if self.future and not self.future.done():
            self.future.set_result("merge")

    @discord.ui.button(
        label="Abort", style=discord.ButtonStyle.red, custom_id="afk_abort"
    )
    async def abort_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        self.action = "abort"
        self._disable_all_buttons()
        if interaction.message.embeds:
            embed = interaction.message.embeds[0]
            embed.color = 0xEF4444
            embed.title = f"Task Aborted: {self.task_id}"
            embed.add_field(
                name="Action",
                value=f"Aborted by **{interaction.user.display_name}**",
                inline=False,
            )
            await interaction.response.edit_message(embed=embed, view=self)
        if self.future and not self.future.done():
            self.future.set_result("abort")

    def _disable_all_buttons(self):
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True
