"""Small deterministic command parser; no network or LLM required."""
import re


class CommandParser:
    def parse(self, text):
        normalized = re.sub(r"[^a-z0-9 ]", " ", (text or "").lower()).strip()
        normalized = " ".join(normalized.split())
        if not normalized:
            return {"intent": "unknown"}

        stop_phrases = (
            "stop guidance", "stop speaking", "be quiet", "stop navigation",
            "cancel find", "stop finding",
        )
        if any(phrase in normalized for phrase in stop_phrases):
            return {"intent": "stop"}

        resume_phrases = ("resume guidance", "start guidance", "continue guidance")
        if any(phrase in normalized for phrase in resume_phrases):
            return {"intent": "resume"}
        if "repeat" in normalized:
            return {"intent": "repeat"}

        scene_phrases = (
            "path clear", "obstacle ahead", "in front of me", "what is ahead",
            "what is in front", "describe the scene", "describe scene",
            "what is around me", "what is around", "tell me what you see",
            "what do you see", "describe surroundings", "describe the surroundings",
            "describe room", "describe the room", "what do i see",
        )
        if any(phrase in normalized for phrase in scene_phrases):
            return {"intent": "scene"}
        if any(phrase in normalized for phrase in ("how far", "distance to", "nearest object")):
            return {"intent": "distance"}

        match = re.search(
            r"(?:find|locate|where is|where are|look for) (?:my |the |a |an )?(.+)",
            normalized,
        )
        if match:
            target = match.group(1).strip()
            if target:
                return {"intent": "find", "target": target}
        return {"intent": "unknown", "text": normalized}
