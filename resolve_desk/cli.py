import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["serve"])
    parser.add_argument("--port", type=int, default=8771)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run("resolve_desk.api:app", host="127.0.0.1", port=args.port)
