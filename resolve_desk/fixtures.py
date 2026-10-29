import json
from pathlib import Path


def load_demo_cases():
    return json.loads((Path(__file__).parent / "fixtures/validation.json").read_text())


def load_v2_demo_cases():
    return json.loads((Path(__file__).parent / "fixtures/validation-v2.json").read_text())


def load_v3_demo_cases():
    return json.loads((Path(__file__).parent / "fixtures/validation-v3.json").read_text())


def load_v4_demo_cases():
    return json.loads((Path(__file__).parent / "fixtures/validation-v4.json").read_text())


def load_v5_demo_cases():
    return json.loads((Path(__file__).parent / "fixtures/validation-v5.json").read_text())


def load_v6_demo_cases():
    return json.loads((Path(__file__).parent / "fixtures/validation-v6.json").read_text())


def load_v6_1_demo_cases():
    return json.loads((Path(__file__).parent / "fixtures/validation-v6_1.json").read_text())


def load_v7_demo_cases():
    return json.loads((Path(__file__).parent / "fixtures/validation-v7.json").read_text())
