#!/usr/bin/env python3

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from bitbots_utils import game_settings  # noqa

if __name__ == "__main__":
    game_settings.main()
