# src/utils.py

import time
import random

from src.config import MSEC, SECS_IN_MIN

# ==================================
#  HELPER FUNCTIONS
# ==================================

def get_run_id() -> str:
    """ Generates a unique ID for the current run. """
    return str(random.randint(10000, 99999))

def start_timer() -> float:
    """
    Start a timer
    """
    return time.time()

def get_time(start_time_float: float) -> str:
    diff = abs(time.time() - start_time_float)
    _, remainder = divmod(diff, SECS_IN_MIN*SECS_IN_MIN)
    minutes, seconds = divmod(remainder, SECS_IN_MIN)
    fractional_seconds = seconds - int(seconds)

    ms = fractional_seconds * MSEC
    return f"{int(minutes)}m {int(seconds)}s {int(ms)}ms"

def show_timer(start_time_int: float) -> str:
    print(f"Run Time: {get_time(start_time_int)}")

def show_banner(title: str, section: str = '') -> None:
    """Prints a stylized banner for console readability."""
    padding = 4
    strlen = len(title) + padding
    line = '+-' + '-' * strlen + '-+'

    print('')
    print(line)
    print('|  ' + title.upper() + '  |')
    print(line)

    if section:
        print('| ' + section)

    print('')
