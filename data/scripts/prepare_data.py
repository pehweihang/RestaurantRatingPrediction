import argparse
import os
from sklearn.preprocessing import LabelEncoder

import pandas as pd

USER_ID = "user_id"
RESTAURANT_ID = "restaurant_id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file_path")
    parser.add_argument("output_file_dir")
    parser.add_argument("-mur", "--min_user_reviews", default=6)
    parser.add_argument("-mrr", "--min_restaurant_reviews", default=6)
    parser.add_argument("-ts", "--test_samples", default=1)

    return parser.parse_args()


def main(
    input_file_path: str,
    output_file_dir: str,
    min_user_reviews: int,
    min_restaurant_reviews: int,
) -> None:
    os.makedirs(output_file_dir, exist_ok=True)
    df = pd.read_csv(input_file_path)
    while (
        df[USER_ID].value_counts().min() < min_user_reviews
        or df[RESTAURANT_ID].value_counts().min() < min_restaurant_reviews
    ):
        df = df[
            df[USER_ID].isin(
                df[USER_ID]
                .value_counts()[lambda x: x >= min_user_reviews]
                .index
            )
        ]
        df = df[
            df[RESTAURANT_ID].isin(
                df[RESTAURANT_ID]
                .value_counts()[lambda x: x >= min_user_reviews]
                .index
            )
        ]
    df.to_csv(os.path.join(output_file_dir, "full.csv"), index=False)

    # Convert string ids to numbers
    user_id_encoder = LabelEncoder()
    user_id_encoder.fit(df[USER_ID])
    df[USER_ID] = user_id_encoder.transform(df[USER_ID])
    restaurant_id_encoder = LabelEncoder()
    restaurant_id_encoder.fit(df[RESTAURANT_ID])
    df[RESTAURANT_ID] = restaurant_id_encoder.transform(df[RESTAURANT_ID])

    test = (
        df.groupby(USER_ID)
        .apply(pd.DataFrame.sample, n=1)
        .reset_index(level=0, drop=True)
    )
    train = df[~df.index.isin(test.index)]
    train.to_csv(os.path.join(output_file_dir, 'train.csv'), index=False)
    test.to_csv(os.path.join(output_file_dir, 'test.csv'), index=False)


if __name__ == "__main__":
    args = parse_args()
    main(
        args.input_file_path,
        args.output_file_dir,
        args.min_user_reviews,
        args.min_restaurant_reviews,
    )
