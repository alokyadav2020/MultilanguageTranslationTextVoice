import importlib
mods=['fastapi','sqlalchemy','passlib','jose','pydantic']
print({m: bool(importlib.util.find_spec(m)) for m in mods})
