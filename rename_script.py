import glob
import os
import shutil
import sys

REQUIRED_DIR = str(sys.argv[1])
list_files = glob.glob(os.path.join(REQUIRED_DIR, '*.png'))
count = 0
tmp_dir = os.path.join(os.getcwd(), REQUIRED_DIR + '_tmp')
print(tmp_dir)
if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)

for filename in list_files:
    source_file = os.path.join(os.getcwd(), filename)
    renamed_file = os.path.join(os.getcwd(), tmp_dir, str(count) + '.png')
    count += 1
    print(f'Rename file {source_file} to {renamed_file}')
    os.rename(source_file, renamed_file)

print(f'Renamed {count} files successfully')

shutil.rmtree(REQUIRED_DIR)
print(f'Delete {REQUIRED_DIR} dir successfully')

os.rename(tmp_dir, os.path.join(os.getcwd(), REQUIRED_DIR))
print(f'Rename dir {tmp_dir} to {REQUIRED_DIR}')
