import asyncio
import logging
import signal
import sys

from maulness.config import config
from maulness.storage.db import StorageManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("maulness.daemon")


async def main():
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
    discord_task = None
    if config.has_discord:
        logger.info("Discord credentials detected. Starting Discord Gateway...")
        from maulness.discord.bot import MaulnessBot

        bot = MaulnessBot(storage=storage)
        discord_task = asyncio.create_task(bot.start(config.discord_bot_token))
    else:
        logger.warning(
            "No Discord credentials found in ~/.config/maulness/env. "
            "Daemon running in CLI/background standby mode."
        )

    logger.info("Maulness daemon is ready and listening.")

    # Await stop event
    await stop_event.wait()

    if discord_task and not discord_task.done():
        logger.info("Closing Discord connection...")
        discord_task.cancel()
        try:
            await discord_task
        except asyncio.CancelledError:
            pass

    logger.info("Maulness daemon terminated cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
