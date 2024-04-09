import cProfile
import pstats
import io
import functools
from typing import Any
from unittest.mock import MagicMock
import os


class DeleteDatabaseAfterTest:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        return self.db

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.delete_db()
        if exc_type:
            if exc_type is AssertionError:
                raise AssertionError(f"{exc_val} (Source: {exc_tb.tb_frame.f_code.co_filename}:{exc_tb.tb_lineno})")
            raise exc_val

class StrictMock(MagicMock):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if 'return_value' not in self.__dict__ and 'side_effect' not in self.__dict__:
            raise AttributeError(f"'{type(self).__name__}' object's '{self._mock_name}' method is not defined")
        else:
            return super().__call__(*args, **kwargs)


def profile_callstack(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return result
    return wrapper

def assert_env_var_present(env_var: str):
    return env_var in os.environ and os.environ[env_var] != ""