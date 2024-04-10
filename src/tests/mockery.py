from unittest.mock import MagicMock
from typing import Any
import sqlite3
import pickle
import hashlib
import os
import inspect
from collections import defaultdict


class StrictMock(MagicMock):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if "return_value" not in self.__dict__ and "side_effect" not in self.__dict__:
            raise AttributeError(
                f"'{type(self).__name__}' object's '{self._mock_name}' method is not defined"
            )
        else:
            return super().__call__(*args, **kwargs)


class HyperMock(MagicMock):
    def __init__(self, serialization_dir: str):
        super().__init__()
        self._db = ObjectDatabase(serialization_dir)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        pass


class HyperMockRecorder:
    def __init__(
        self, obj, object_chain="", object_db_dir="", test_name="", object_db=None
    ):
        # Store the object to be wrapped
        self._wrapped_obj = obj
        self._wrapped_attr_cache = {}
        if object_chain:
            self._object_chain = object_chain + "." + obj.__class__.__name__
        else:
            self._object_chain = obj.__class__.__name__
        self._function_call_accounting = defaultdict(lambda: 0)

        if object_db is None:
            if object_db_dir:
                self._object_db = ObjectDatabase(object_db_dir)
        else:
            self._object_db = object_db
        self._test_name = test_name

    def __getattr__(self, name):
        if name in [
            "_wrapped_obj",
            "_wrapped_attr_cache",
            "_object_chain",
            "_function_call_accounting",
            "_object_db",
            "_test_name",
        ]:
            return super().__getattribute__(name)

        # Intercept attribute access
        attr = getattr(self._wrapped_obj, name)
        print(f"Accessing attribute '{name}'")

        # Check if the attribute is an object or a class that should be wrapped
        if self.is_real_method(attr) or self.is_function(attr):

            def wrapper(*args, **kwargs):
                function_fqn = f"{self._object_chain}.{name}"
                print(
                    f"Calling {function_fqn} for the {self._function_call_accounting[name]} time with args {args} and kwargs {kwargs}"
                )
                self._function_call_accounting[name] += 1
                result = attr(*args, **kwargs)
                print(f"'{self._object_chain}.{name}' returned {result}")
                self._object_db.store_function_call(
                    self._test_name, function_fqn, self._function_call_accounting[name], result, *args, **kwargs
                )
                return result

            return wrapper
        else:
            if name not in self._wrapped_attr_cache:
                self._wrapped_attr_cache[name] = HyperMockRecorder(
                    attr,
                    self._object_chain,
                    object_db=self._object_db,
                    test_name=self._test_name,
                )
            return self._wrapped_attr_cache[name]

    def __setattr__(self, name, value):
        # Intercept attribute assignment, except for the wrapped object itself
        if name in [
            "_wrapped_obj",
            "_wrapped_attr_cache",
            "_object_chain",
            "_function_call_accounting",
            "_object_db",
            "_test_name",
        ]:
            super().__setattr__(name, value)
        else:
            print(f"Setting attribute '{name}' to {value}")
            setattr(self._wrapped_obj, name, value)

    def __delattr__(self, name):
        # Intercept attribute deletion
        print(f"Deleting attribute '{name}'")
        delattr(self._wrapped_obj, name)

    def __call__(self, *args, **kwargs):
        # Intercept calls if the object is callable
        print(
            f"Calling object {self._object_chain} for the {self._function_call_accounting['self']} time with args {args} and kwargs {kwargs}"
        )
        self._function_call_accounting["self"] += 1
        return self._wrapped_obj(*args, **kwargs)

    @staticmethod
    def is_real_method(attr):
        return inspect.ismethod(attr) or (
            callable(attr) and inspect.isfunction(attr) and hasattr(attr, "__self__")
        )

    @staticmethod
    def is_function(attr):
        return callable(attr) and inspect.isfunction(attr)

    @staticmethod
    def is_callable_object(attr):
        return (
            callable(attr)
            and not inspect.isfunction(attr)
            and not inspect.ismethod(attr)
        )


class ObjectDatabase:
    def __init__(self, serialization_dir: str):
        self.db_path = f"{serialization_dir}/object_database.db"
        os.makedirs(serialization_dir, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS function_calls
                               (test_name TEXT, function_name TEXT, call_number INT, combined_hash TEXT, returned_value BLOB)"""
        )
        self.conn.commit()

    def _pickle_and_hash(self, *args, **kwargs) -> str:
        pickled_args = pickle.dumps(args)
        pickled_kwargs = pickle.dumps(kwargs)
        combined_pickle = pickled_args + pickled_kwargs
        hash_object = hashlib.sha256(combined_pickle)
        return hash_object.hexdigest()

    def store_function_call(
        self,
        test_name: str,
        function_name: str,
        call_number: int,
        returned_values: Any,
        *args,
        **kwargs,
    ):
        combined_hash = self._pickle_and_hash(*args, **kwargs)
        pickled_returned_values = pickle.dumps(returned_values)
        self.cursor.execute(
            """INSERT INTO function_calls (test_name, function_name, call_number, combined_hash, returned_value)
                               VALUES (?, ?, ?, ?, ?)""",
            (
                test_name,
                function_name,
                call_number,
                combined_hash,
                pickled_returned_values,
            ),
        )
        self.conn.commit()

    def find_function_call(
        self, test_name: str, function_name: str, call_number: int, *args, **kwargs
    ) -> Any:
        combined_hash = self._pickle_and_hash(*args, **kwargs)
        self.cursor.execute(
            """SELECT returned_value FROM function_calls WHERE test_name=? AND function_name=? AND call_number=? AND combined_hash=?""",
            (test_name, function_name, call_number, combined_hash),
        )
        result = self.cursor.fetchone()
        if result:
            return pickle.loads(result[0])
        else:
            return None

    def __del__(self):
        self.conn.close()
