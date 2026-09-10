import asyncio
import logging
import signal
import sys
from typing import Optional

from maulness.config import config
from maulness.storage.db import StorageManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("maulness.daemon")


async def main(profile_filter: Optional[str] = None):
    logger.info("Starting Maulness daemon service...")

    # Initialize storage
    storage = StorageManager(db_path=config.db_path)
    await storage.initialize()
    logger.info("SQLite storage initialized at %s", config.db_path)

    # Setup termination events
    stop_event = asyncio.Event()

    def handle_signal():
        logger.info("Received termination signal. Shutting down...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_signal)
        except NotImplementedError:
            pass

    # Discord Gateway integration
    discord_tasks: list[asyncio.Task] = []
    bots = []
    from maulness.core.profiles import ProfileManager
    from maulness.discord.bot import MaulnessBot

    pm = ProfileManager()
    all_profiles = pm.list_profiles()

    if profile_filter:
        target_profile = pm.get_profile(profile_filter)
        token = target_profile.env_vars.get("DISCORD_BOT_TOKEN") or config.discord_bot_token
        if token:
            logger.info("Starting Discord Gateway for profile '%s'...", target_profile.name)
            bot = MaulnessBot(storage=storage, profile=target_profile)
            bots.append(bot)
            discord_tasks.append(asyncio.create_task(bot.start(token)))
        else:
            logger.warning("No Discord token found for profile '%s'.", profile_filter)

    elif config.gateway_multiplex_profiles:
        logger.info("Gateway multiplexing enabled (gateway.multiplex_profiles: true)")
        allowlist = config.gateway_multiplex_profile_allowlist
        served_profiles = [
            p for p in all_profiles
            if not allowlist or p.name in allowlist
        ]
        logger.info("Multiplexing profiles: %s", [p.name for p in served_profiles])

        # Group profiles by token to deduplicate client connections
        token_groups: dict[str, list[Any]] = {}
        for p in served_profiles:
            token = p.env_vars.get("DISCORD_BOT_TOKEN")
            if not token and p.name == "default":
                token = config.discord_bot_token

            if token:
                token_groups.setdefault(token, []).append(p)
            else:
                logger.debug("Profile '%s' has no DISCORD_BOT_TOKEN configured, skipping adapter.", p.name)

        for token, group in token_groups.items():
            profile_names = [p.name for p in group]
            logger.info("Spawning Discord bot adapter for profile(s) %s...", profile_names)
            bot = MaulnessBot(storage=storage, profiles=group)
            bots.append(bot)
            discord_tasks.append(asyncio.create_task(bot.start(token)))

        if not discord_tasks:
            logger.warning(
                "Multiplexing active but no profile defines DISCORD_BOT_TOKEN in .env. "
                "Daemon running in standby mode."
            )
    else:
        if config.has_discord:
            logger.info("Starting default Discord Gateway...")
            bot = MaulnessBot(storage=storage, profile_name="default")
            bots.append(bot)
            discord_task = asyncio.create_task(bot.start(config.discord_bot_token))
            discord_tasks.append(discord_task)
        else:
            logger.warning(
                "No Discord credentials found in ~/.config/maulness/env. "
                "Daemon running in CLI/background standby mode."
            )

    logger.info("Maulness daemon is ready and listening (%d bot adapters active).", len(discord_tasks))

    # Await stop event
    await stop_event.wait()

    for bot, task in zip(bots, discord_tasks):
        if not task.done():
            p_names = [p.name for p in bot.profiles] if hasattr(bot, "profiles") else [bot.profile_name]
            logger.info("Closing Discord connection for profile(s) %s...", p_names)
            await bot.close()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    logger.info("Maulness daemon terminated cleanly.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Maulness Daemon Service")
    parser.add_argument("--profile", "-p", default=None, help="Target a specific profile")
    args = parser.parse_args()

    asyncio.run(main(profile_filter=args.profile))
