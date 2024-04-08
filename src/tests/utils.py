import cProfile
import pstats
import io
import functools


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