import asyncio
import logging
from pathlib import Path
from typing import Optional

from maulness.config import config
from maulness.core.models import AgentMessageEvent
from maulness.core.profiles import Profile, ProfileManager

logger = logging.getLogger("maulness.memory.extractor")

EXTRACTION_SYSTEM_PROMPT = """You are an automated memory librarian for Maul's personal AI agent harness (Maulness).
Analyze the conversation transcript provided.
Extract ONLY durable developer preferences, system constraints, or architectural decisions explicitly stated or decided.
Examples of durable facts to extract:
- User preferences (e.g. "Prefers service-name addressing over host ports", "Uses pnpm instead of npm")
- Tech stack facts (e.g. "expense-tracker uses Laravel 12 + Inertia / Vue 3")
- Architectural rules (e.g. "Never push to remote without explicit approval")

DO NOT extract temporary conversation topics, transient debug logs, code snippets, greetings, or one-off questions.
If no durable facts or rules are present in the transcript, reply ONLY with: NONE
If durable facts are found, output them as concise markdown bullet points (one per line, starting with '- ')."""


class AutonomousMemoryExtractor:
    """Background worker that autonomously extracts durable facts from conversations into MEMORY.md."""

    def __init__(self, memory_file: Optional[Path] = None):
        self.memory_file = memory_file or (config.config_dir / "MEMORY.md")
        self.profile_manager = ProfileManager()

    async def extract_and_update(
        self,
        messages: list[dict[str, str]],
        profile_name: str = "default",
    ) -> list[str]:
        """Asynchronously analyze conversation history and record newly discovered facts."""
        if len(messages) < 4:
            return []

        # Format conversation transcript
        transcript_lines = []
        for m in messages[-10:]:
            role = m.get("role", "user").upper()
            content = m.get("content", "").strip()
            if content:
                transcript_lines.append(f"{role}: {content[:500]}")

        transcript_text = "\n".join(transcript_lines)
        if not transcript_text.strip():
            return []

        try:
            # Instantiate a fast extraction provider (gemini-3.6-flash or current profile fallback)
            from maulness.core.providers.factory import get_provider_for_profile

            extract_profile = Profile(
                identity={"name": "memory-librarian"},
                agent={"provider": "gemini", "model": "gemini-3.6-flash", "temperature": 0.1},
            )
            # Inherit API keys from profile or config
            active_p = self.profile_manager.get_profile(profile_name)
            extract_profile.env_vars = active_p.env_vars.copy()

            provider = get_provider_for_profile(extract_profile)

            prompt = f"CONVERSATION TRANSCRIPT:\n{transcript_text}\n\nExtract durable facts according to your instructions:"
            result = await asyncio.wait_for(
                provider.run(session_id="memory_extract", prompt=prompt),
                timeout=25.0,
            )

            result = result.strip()
            if not result or result == "NONE" or "NONE" in result.splitlines()[:1]:
                return []

            # Extract bullet lines
            new_bullets = [
                line.strip() for line in result.splitlines()
                if line.strip().startswith(("-", "*")) and len(line.strip()) > 5
            ]

            if not new_bullets:
                return []

            added = self._append_facts_to_memory(new_bullets)
            if added:
                logger.info("AutonomousMemoryExtractor added %d facts to %s", len(added), self.memory_file)
            return added

        except Exception as e:
            logger.debug("Autonomous memory extraction skipped or failed: %s", e)
            return []

    def _append_facts_to_memory(self, facts: list[str]) -> list[str]:
        """Append new non-duplicate facts to MEMORY.md."""
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        current_content = self.memory_file.read_text(encoding="utf-8") if self.memory_file.exists() else "# Workspace Memory\n"

        existing_lines_lower = {l.strip().lower() for l in current_content.splitlines() if l.strip()}

        facts_to_add = []
        for fact in facts:
            clean_fact = fact.strip()
            if clean_fact.lower() not in existing_lines_lower:
                facts_to_add.append(clean_fact)
                existing_lines_lower.add(clean_fact.lower())

        if not facts_to_add:
            return []

        section_header = "\n## Auto-Learned Knowledge & Decisions\n"
        if section_header.strip() not in current_content:
            updated_content = current_content.rstrip() + "\n" + section_header + "\n".join(facts_to_add) + "\n"
        else:
            updated_content = current_content.rstrip() + "\n" + "\n".join(facts_to_add) + "\n"

        self.memory_file.write_text(updated_content, encoding="utf-8")
        return facts_to_add
