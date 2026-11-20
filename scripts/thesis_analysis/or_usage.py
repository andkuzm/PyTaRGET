"""Total OpenRouter account usage in USD (prints one number; the key is read from ~/.config/openrouter/key and never printed)."""
import json, os, urllib.request
key = open(os.path.expanduser("~/.config/openrouter/key")).read().strip()
req = urllib.request.Request("https://openrouter.ai/api/v1/credits", headers={"Authorization": "Bearer " + key, "User-Agent": "python-urllib"})
print(round(json.load(urllib.request.urlopen(req, timeout=60))["data"]["total_usage"], 5))
