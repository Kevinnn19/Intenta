from torch.fx.experimental.unification.multipledispatch.dispatcher import source

from app.routing.intent_router import IntentRouter

router = IntentRouter()

test_intents = [
    "model_saving",
    "refund",
    "login_issue",
    "unknown_intent"
]

for intent in test_intents:
    source = router.route(intent)

    print(f"Intent: {intent}")
    print(f"Source: {source}")
    print("-" * 30)