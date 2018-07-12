import os
import pandas as pd
import fnmatch

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
            new_path = os.path.join(parent, filename + " v_{:02d}".format(n) + file_extension)
            if os.path.exists(new_path):
                n+=1
            else:
                break
        return new_path
    else:
        return full_path

def find_files(base, pattern):
    '''Return list of files matching pattern in base folder.'''
    #matching_files = [n for n in fnmatch.filter(os.listdir(base), pattern) if os.path.isfile(os.path.join(base, n))]
    matching_files_and_folders = fnmatch.filter(os.listdir(base), pattern)
    return len(matching_files_and_folders)>0

def createLogDir(name = "", force_numerical_ordering = True):
    n = 1

    # Add padding
    if name != "" and name[0] != " ":
        name = " " + name

    # Check for existence
    basepath = "./tf_logs"
    if not os.path.exists(basepath):
        os.mkdir(basepath)

    if force_numerical_ordering:
        while find_files(basepath, str(n) + " *") or os.path.exists(os.path.join(basepath, str(n)    )) :
            n += 1
    else:
        while os.path.exists(os.path.join(basepath, str(n) + name )):
            n += 1

    # Create
    logdir = os.path.join(basepath, str(n) + name)
    os.mkdir(logdir)
    training_accuracy_list = []
    print(logdir)
    return logdir


if __name__=='__main__':
    pass