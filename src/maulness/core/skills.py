from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml

from maulness.config import config

USER_SKILLS_DIR = config.skills_dir
TEMPLATE_SKILLS_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "templates" / "skills"
)


@dataclass
class Skill:
    name: str
    description: str
    path: Path
    instruction: str


class SkillManager:
    """Manages discovery and composition of root and profile-level skills."""

    def __init__(
        self,
        root_skills_dir: Optional[Path] = None,
        template_skills_dir: Optional[Path] = None,
    ):
        self.root_skills_dir = root_skills_dir or USER_SKILLS_DIR
        self.template_skills_dir = (
            template_skills_dir
            if template_skills_dir is not None
            else TEMPLATE_SKILLS_DIR
        )

    def discover_skills_in_dir(self, directory: Path) -> dict[str, Skill]:
        """Scan a directory for skill subdirectories containing SKILL.md."""
        skills: dict[str, Skill] = {}
        if not directory.exists() or not directory.is_dir():
            return skills

        for entry in sorted(directory.iterdir()):
            if entry.is_dir():
                skill_md = entry / "SKILL.md"
                if skill_md.exists():
                    skill = self._parse_skill_file(skill_md, entry)
                    if skill:
                        skills[skill.name] = skill
        return skills

    def list_skills(
        self,
        profile_skills_dir: Optional[Path] = None,
        workspace_path: Optional[Path] = None,
    ) -> dict[str, Skill]:
        """List all effective skills, merging templates, user root, profile, and workspace skills."""
        skills: dict[str, Skill] = {}

        # 1. Built-in template skills
        if self.template_skills_dir and self.template_skills_dir.exists():
            skills.update(self.discover_skills_in_dir(self.template_skills_dir))

        # 2. Root user skills (~/.config/maulness/skills)
        if self.root_skills_dir.exists():
            skills.update(self.discover_skills_in_dir(self.root_skills_dir))

        # 3. Profile-level extending skills (~/.config/maulness/profiles/<name>/skills)
        if profile_skills_dir and profile_skills_dir.exists():
            skills.update(self.discover_skills_in_dir(profile_skills_dir))

        # 4. Workspace-level skills (.maulness/skills or .agents/skills)
        if workspace_path:
            ws = Path(workspace_path).resolve()
            for ws_skill_dir in [
                ws / ".maulness" / "skills",
                ws / ".agents" / "skills",
            ]:
                if ws_skill_dir.exists():
                    skills.update(self.discover_skills_in_dir(ws_skill_dir))

        return skills

    def get_skill(
        self,
        name: str,
        profile_skills_dir: Optional[Path] = None,
        workspace_path: Optional[Path] = None,
    ) -> Optional[Skill]:
        """Retrieve a specific skill definition by name."""
        skills = self.list_skills(
            profile_skills_dir=profile_skills_dir, workspace_path=workspace_path
        )
        return skills.get(name)

    def get_skill_instruction(
        self,
        name: str,
        profile_skills_dir: Optional[Path] = None,
        workspace_path: Optional[Path] = None,
    ) -> str:
        """Load full playbook instructions for a skill (progressive disclosure)."""
        skill = self.get_skill(
            name, profile_skills_dir=profile_skills_dir, workspace_path=workspace_path
        )
        if not skill:
            available = (
                ", ".join(self.list_skills(profile_skills_dir, workspace_path).keys())
                or "none"
            )
            return f"Error: Skill '{name}' not found. Available skills: {available}"

        return (
            f"# Playbook Instruction: {skill.name}\n"
            f"Description: {skill.description}\n"
            f"Location: {skill.path}\n\n"
            f"{skill.instruction}"
        )

    def get_skills_paths(
        self,
        profile_skills_dir: Optional[Path] = None,
        workspace_path: Optional[Path] = None,
    ) -> list[str]:
        """Return unique directory paths containing skills for Antigravity LocalAgentConfig."""
        paths: list[str] = []
        if self.template_skills_dir and self.template_skills_dir.exists():
            paths.append(str(self.template_skills_dir))
        if self.root_skills_dir.exists():
            paths.append(str(self.root_skills_dir))
        if profile_skills_dir and profile_skills_dir.exists():
            paths.append(str(profile_skills_dir))
        if workspace_path:
            ws = Path(workspace_path).resolve()
            for ws_skill_dir in [
                ws / ".maulness" / "skills",
                ws / ".agents" / "skills",
            ]:
                if ws_skill_dir.exists():
                    paths.append(str(ws_skill_dir))
        return list(dict.fromkeys(paths))

    def format_skills_summary(self, skills: dict[str, Skill]) -> str:
        """Format a lightweight progressive disclosure index of available skills for prompt injection."""
        if not skills:
            return ""

        lines = [
            "\n---\n## Available Skills (Progressive Disclosure)",
            "Call tool `load_skill(name)` to load comprehensive instructions when executing a task requiring these playbooks:",
        ]
        for name, skill in skills.items():
            desc = skill.description or "No description provided."
            lines.append(f"- **{name}**: {desc}")

        return "\n".join(lines)

    def _parse_skill_file(self, skill_file: Path, skill_dir: Path) -> Optional[Skill]:
        """Parse SKILL.md with optional YAML frontmatter."""
        try:
            content = skill_file.read_text(encoding="utf-8").strip()
            name = skill_dir.name
            description = ""
            instruction = content

            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = yaml.safe_load(parts[1])
                    if isinstance(frontmatter, dict):
                        name = frontmatter.get("name", name)
                        description = frontmatter.get("description", "")
                    instruction = parts[2].strip()

            return Skill(
                name=name,
                description=description,
                path=skill_dir,
                instruction=instruction,
            )
        except Exception:
            return None
