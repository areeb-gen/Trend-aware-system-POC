from . import retrieve_trends, search_web

_REGISTRY = [retrieve_trends, search_web]

TOOLS = [m.SCHEMA for m in _REGISTRY]

_EXECUTORS = {m.SCHEMA["function"]["name"]: m.execute for m in _REGISTRY}


def dispatch(name: str, args: dict):
    executor = _EXECUTORS.get(name)
    if executor is None:
        raise ValueError(f"Unknown tool: {name}")
    return executor(**args)
