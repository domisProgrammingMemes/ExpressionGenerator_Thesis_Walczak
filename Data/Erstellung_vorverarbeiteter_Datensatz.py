import pandas as pd
import os.path

csv_read_path = r"raw"
csv_write_path = r"preprocessed\\"


if __name__ == "__main__":

    csv_list = os.listdir(csv_read_path)
    number_of_csv_files = len(csv_list)

    for csv in csv_list:
        df = pd.read_csv(csv_read_path + "/" + csv)
        df.sort_values(by="Frame", ascending=True, inplace=True)
        df.drop_duplicates(subset="Frame", inplace=True, ignore_index=True)

        df.to_csv(path_or_buf=csv_write_path + csv, index=False)