from app.routing.routing_config import INTENT_TO_SOURCE

class IntentRouter:
    def __init__(self):
        self.mapping = INTENT_TO_SOURCE

    def route(self, intent: str):
        source = self.mapping.get(intent, "general")
        return source