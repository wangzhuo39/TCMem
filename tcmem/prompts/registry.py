from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_PROMPT_RESOURCE = "default_prompts.yaml"
TEXT_PROMPT_RESOURCES = {
    "entity_extraction": "entity_extraction.txt",
    "task_routing": "task_routing.txt",
    "query_routing": "query_routing.txt",
    "task_metadata_refresh": "task_metadata_refresh.txt",
}


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    system: str
    user: str


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    system_prompt: str
    user_prompt: str


class PromptRegistry:
    def __init__(self, prompts: Mapping[str, PromptTemplate]) -> None:
        self._prompts = dict(prompts)

    @classmethod
    def default(cls) -> "PromptRegistry":
        package_files = resources.files("tcmem.prompts")
        prompts = cls.from_yaml_text(package_files.joinpath(DEFAULT_PROMPT_RESOURCE).read_text(encoding="utf-8"))._prompts
        prompts = dict(prompts)
        for prompt_name, resource_name in TEXT_PROMPT_RESOURCES.items():
            candidate = package_files.joinpath(resource_name)
            if candidate.is_file():
                prompts[prompt_name] = PromptTemplate(system="", user=candidate.read_text(encoding="utf-8").rstrip())
        return cls(prompts)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PromptRegistry":
        if path:
            return cls.from_path(path)
        return cls.default()

    @classmethod
    def from_path(cls, path: str | Path) -> "PromptRegistry":
        prompt_path = Path(path)
        if prompt_path.is_dir():
            prompts = dict(cls.default()._prompts)
            yaml_path = prompt_path / DEFAULT_PROMPT_RESOURCE
            if yaml_path.exists():
                prompts.update(cls.from_path(yaml_path)._prompts)
            for prompt_name, file_name in TEXT_PROMPT_RESOURCES.items():
                txt_path = prompt_path / file_name
                if txt_path.exists():
                    prompts[prompt_name] = PromptTemplate(system="", user=txt_path.read_text(encoding="utf-8").rstrip())
            return cls(prompts)
        if prompt_path.suffix.lower() in {".yaml", ".yml"}:
            return cls.from_yaml_text(prompt_path.read_text(encoding="utf-8"))
        if prompt_path.suffix.lower() == ".txt":
            return cls.from_mapping({prompt_path.stem: {"system": "", "user": prompt_path.read_text(encoding="utf-8")}})
        raise ValueError(f"unsupported prompt file type: {prompt_path}")

    @classmethod
    def from_yaml_text(cls, text: str) -> "PromptRegistry":
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError("prompt file must contain a mapping of prompt names")
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PromptRegistry":
        prompts: dict[str, PromptTemplate] = {}
        for name, raw_template in data.items():
            if not isinstance(raw_template, Mapping):
                raise ValueError(f"prompt {name!r} must be a mapping")
            system = raw_template.get("system")
            user = raw_template.get("user")
            if not isinstance(system, str) or not isinstance(user, str):
                raise ValueError(f"prompt {name!r} must include string fields: system, user")
            prompts[str(name)] = PromptTemplate(system=system.strip(), user=user.rstrip())
        return cls(prompts)

    def render(self, name: str, **values: Any) -> RenderedPrompt:
        template = self._prompts.get(name)
        if template is None:
            template = self.default()._prompts.get(name)
        if template is None:
            raise KeyError(f"unknown prompt template: {name}")
        user_prompt = template.user
        for key, value in values.items():
            user_prompt = user_prompt.replace("{{" + key + "}}", str(value))
        return RenderedPrompt(system_prompt=template.system, user_prompt=user_prompt)
