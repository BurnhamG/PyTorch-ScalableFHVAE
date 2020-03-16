import os
import sys


def maybe_makedir(d):
    try:
        os.makedirs(d, exist_ok=True)
    except OSError as e:
        print(e)
        sys.exit(2)
