# config/settings.py
import json
import os

# Resolve the directory that holds the per-country JSON files relative to this
# file, not the shell's working directory. Otherwise the app would only work
# when launched from one specific folder (e.g. `python -m api.main` vs running
# a script from elsewhere), which makes it fragile to run in different ways.
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))


class STIGConfig:
    """
    Reads the per-country configuration that makes STIG multi-country.

    Each country is one JSON file in `config_dir` holding the country name, the
    system prompt the LLM adopts, UI labels, and the Wikipedia topics used to
    build that country's knowledge base. Keeping this in data files rather than
    code means adding a country is a data change, not a code change.
    """

    def __init__(self, config_dir: str):
        # where are the config files stored?
        self.config_dir = config_dir

    def load_country(self, country: str) -> dict:
        """Load and return config for a given country"""
        # 1. build the file path using country name (hint: nigeria -> config/nigeria.json)
        # Normalise to lowercase so "Germany", "germany" and "GERMANY" all resolve
        # to the same file — the country name arrives as a user-controlled string
        # from the API, so casing must not decide whether a lookup succeeds.
        path = f"{self.config_dir}/{country.lower()}.json"

        # 2. check if the file exists.
        # Fail loudly with the list of valid options instead of a bare
        # FileNotFoundError — that way a typo reads as "unknown country" rather
        # than looking like a crash, and the caller can correct it immediately.
        if not os.path.exists(path):
            available = ", ".join(self.available_countries())
            raise FileNotFoundError(
                f"No config found for '{country}'. Available countries: {available}"
            )

        # 3. open and read the JSON file (hint: use open() and json.load())
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # 4. return the config dict
        return config

    def available_countries(self) -> list:
        """Return list of all available countries"""
        # Derive the list from the filesystem instead of a hardcoded list, so
        # dropping in a new <country>.json is all it takes to expose it — e.g.
        # the /countries endpoint the frontend uses to build its country picker.
        filenames = os.listdir(self.config_dir)
        return sorted(
            filename[: -len(".json")]
            for filename in filenames
            if filename.endswith(".json")
        )


if __name__ == "__main__":
    # Standalone sanity check: `python config/settings.py` should print the
    # Nigeria config, proving both methods above actually do something.
    config = STIGConfig(CONFIG_DIR)
    nigeria = config.load_country("nigeria")
    print(f"Loaded country: {nigeria['country']}")
    print(f"Topic count: {len(nigeria['topics'])}")
