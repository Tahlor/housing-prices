import os

def export_data(df, file_path, overwrite = False):
    #pd.to_numeric(df.columns(["Id"]), downcast='int')
    df.Id = df.Id.astype(int)

    output_folder = os.path.dirname(file_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not overwrite:
        file_path = increment_path_version(file_path)
    df.to_csv(file_path, sep=',', encoding='utf-8', index=False)

def increment_path_version(full_path):
    n = 1
    if os.path.exists(full_path):
        (parent, child) = os.path.split(full_path)
        filename, file_extension = os.path.splitext(child)
        while True:
            print("path already exists, incrementing...")
            new_path = os.path.join(parent, filename + " v_{:02d}".format(n) + file_extension)
            if os.path.exists(new_path):
                n+=1
            else:
                break
        return new_path
    else:
        return full_path


if __name__=='__main__':
    pass