"""Memory providers for different frameworks.

Keep package import lightweight so provider-specific heavy dependencies are only
loaded when that provider is actually requested.
"""

from importlib import import_module

__all__ = [
    "AgentKBProvider",
    "SkillWeaverProvider",
    "MobileEProvider",
    "ExpeLProvider",
    "PromptBasedMemoryProvider",
    "SiliconFriendProvider",
]

_PROVIDER_MODULES = {
    "AgentKBProvider": ".agent_kb_provider",
    "SkillWeaverProvider": ".skillweaver_provider",
    "MobileEProvider": ".mobilee_provider",
    "ExpeLProvider": ".expel_provider",
    "PromptBasedMemoryProvider": ".prompt_based_memory_provider",
    "SiliconFriendProvider": ".siliconfriend_provider",
}


def __getattr__(name):
    module_name = _PROVIDER_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, name)
