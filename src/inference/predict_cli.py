"""Subprocess-safe CLI wrapper for model prediction."""

import json
import sys

from src.inference.predictor import predict_price_direct


def main() -> None:
    """Read request features from argv and print prediction JSON."""
    if len(sys.argv) < 2:
        raise ValueError("Expected a JSON-serialized feature payload as the first argument")

    input_data = json.loads(sys.argv[1])
    predicted_price = predict_price_direct(input_data)
    print(json.dumps({"predicted_price": predicted_price}))


if __name__ == "__main__":
    main()
