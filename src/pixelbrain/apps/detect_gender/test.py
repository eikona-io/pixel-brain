import sys
import os
import modal

stub = modal.Stub("example-hello-world")

@stub.function()
def f():
    return os.environ

@stub.local_entrypoint()
def main():
    print(f.remote())

