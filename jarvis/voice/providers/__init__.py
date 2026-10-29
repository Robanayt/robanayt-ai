"""Voice providers shipped with JARVIS.

All of them are local Windows providers; nothing here sends audio anywhere.
Remote speech providers would be a separate decision (network, cost, data
sharing) and would have to respect the same privacy boundary.
"""

from __future__ import annotations

from jarvis.voice.providers.powershell import (
    CommandResult,
    SubprocessPowerShellRunner,
    probe_audio_devices,
)
from jarvis.voice.providers.sapi import SapiSpeechRecognizer, SapiSpeechSynthesizer

__all__ = [
    "CommandResult",
    "SapiSpeechRecognizer",
    "SapiSpeechSynthesizer",
    "SubprocessPowerShellRunner",
    "probe_audio_devices",
]
