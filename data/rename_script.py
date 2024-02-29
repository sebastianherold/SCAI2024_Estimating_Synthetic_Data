import os

def rename_files_in_dir(dir_path):
    for filename in os.listdir(dir_path):
        if filename.endswith('.csv') and "_TVAE" in filename:
            # split the filename on '_TVAE' and take the first part
            new_filename = filename.split('_TVAE')[0] + '.csv'
            src = os.path.join(dir_path, filename)
            dst = os.path.join(dir_path, new_filename)
            os.rename(src, dst) # rename the file

def main():
    # Specify your directory path here
    dir_path = "./synthetic_tvae_D305" 

    rename_files_in_dir(dir_path)

if __name__ == '__main__':
    main()