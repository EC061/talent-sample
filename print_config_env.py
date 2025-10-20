#!/usr/bin/env python3
"""
Print environment variable exports derived from config.yml for shell consumption.

Usage:
  eval "$(python3 print_config_env.py)"
"""

from config_loader import load_config, as_env_dict


def main() -> None:
    cfg = load_config()
    env = as_env_dict(cfg)
    for key, value in env.items():
        if value is None:
            continue
        # Safely quote values for shell
        sval = str(value).replace('"', '\\"')
        print(f'export {key}="{sval}"')


if __name__ == "__main__":
    main()


