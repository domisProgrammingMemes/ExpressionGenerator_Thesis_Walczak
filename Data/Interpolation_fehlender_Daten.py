import pandas as pd
import numpy as np
import os.path

csv_read_path = r"preprocessed"
csv_write_path = r"dataset\\"


if __name__ == "__main__":

    csv_list = os.listdir(csv_read_path)
    number_of_csv_files = len(csv_list)
    step = 1

    for csv in csv_list:
        df = pd.read_csv(csv_read_path + "/" + csv)

        idx = np.arange(df["Frame"].min(), df["Frame"].max() + step, step)
        df = df.set_index("Frame").reindex(idx).reset_index()

        # several interpolation methods
        df.interpolate(method="linear", inplace=True)
        # df.interpolate(method="quadratic", inplace=True)
        # df.interpolate(method="cubic", inplace=True)
        # df.interpolate(method="polynomial", order=2, inplace=True)

        name, type = csv.split(sep=".")
        file = name + "_lfill." + type
        # csv = name + "_qfill." + type
        # csv = name + "_cfill." + type
        # csv = name + "_pfill." + type

        df.to_csv(path_or_buf=csv_write_path + file, index=False)