import argparse
import json

import pandas as pd

JSON_USER_ID_KEY = "user_id"
JSON_RESTAURANT_ID_KEY = "business_id"
JSON_RATING_KEY = "rating"
USER_ID = "user_id"
RESTAURANT_ID = "restaurant_id"
RATING = "rating"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file_path")
    parser.add_argument("output_file_path")
    return parser.parse_args()


def main(input_file_path: str, output_file_path: str):
    f = open(input_file_path)
    df = pd.DataFrame(columns=[USER_ID, RESTAURANT_ID, RATING])
    for i, l in enumerate(f.readlines(), 1):
        x = json.loads(l)
        row = pd.Series(
            {
                USER_ID: x[JSON_USER_ID_KEY],
                RESTAURANT_ID: x[JSON_RESTAURANT_ID_KEY],
                RATING: x[JSON_RATING_KEY],
            }
        )
        df.loc[i] = row
    df.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args.input_file_path, args.output_file_path)
