import os
import glob
import shutil
import argparse
import zipfile
import tqdm


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip_path', type=str)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--unzip_remove', action='store_true', default=False)
    opt = parser.parse_args()
    return opt


def main(opt):
    zip_files = glob.glob(os.path.join(opt.zip_path, '**', '*.zip'), recursive=True)
    for file in tqdm.tqdm(zip_files):
        # target_file = zipfile.ZipFile(file)
        if opt.save_path:
            # print(opt.save_path)
            # print(os.path.dirname(file))
            # print(os.path.basename(file).split('.')[0])
            target_path = os.path.join(opt.save_path, os.path.dirname(file), os.path.basename(file).split('.')[0])
            # print(target_path)
            # target_file.extractall(target_path)
        else:
            target_path = os.path.join(os.path.dirname(file), os.path.basename(file).split('.')[0])
            # target_file.extractall(target_path)

        os.makedirs(target_path, exist_ok=True)
        unzip(file, target_path)
        # target_file.close()
        print(f'{os.path.basename(file)} is released on {target_path}')
        if opt.unzip_remove:
            os.remove(file)


def unzip(source_file, dest_path):
    with zipfile.ZipFile(source_file, 'r') as zf:
        zipInfo = zf.infolist()
        for member in zipInfo:
            try:
                # print(member.filename.encode('cp437').decode('euc-kr', 'ignore'))
                # os.makedirs(os.path.join(dest_path, member.filename.encode('cp437').decode('euc-kr', 'ignore')), exist_ok=True)
                member.filename = member.filename.encode('utf-8').decode('euc-kr', 'ignore')
                # member.filename = member.filename.encode('cp437').decode('euc-kr', 'ignore')
                zf.extract(member, dest_path)
            except:
                print(source_file)
                raise Exception('what?!')


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)