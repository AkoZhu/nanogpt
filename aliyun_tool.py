import aligo
from aligo import Aligo
import argparse
refresh_token = ""
ali = Aligo(refresh_token=refresh_token)



def down_file_or_folder(remote_path, local_folder, is_file=False):
    file = (
        ali.get_file_by_path(remote_path)
        if is_file
        else ali.get_folder_by_path(remote_path)
    )

    if is_file:
        ali.download_file(file_id=file.file_id, local_folder=local_folder)
    else:
        ali.download_folder(folder_file_id=file.file_id, local_folder=local_folder)


def upload_file_or_folder(local_file_folder, remote_folder, is_file=False):
    remote_folder_id = ali.get_folder_by_path(remote_folder).file_id

    if is_file:
        ali.upload_file(file_path=local_file_folder, parent_file_id=remote_folder_id)
    else:
        ali.upload_folder(
            folder_path=local_file_folder, parent_file_id=remote_folder_id
        )


def main():
    parser = argparse.ArgumentParser(
        description="Download file or folder from Aliyun Drive. 默认下载 & 文件夹"
    )
    parser.add_argument("-up", "--is_up", action="store_true", help="默认是下载模式")
    parser.add_argument(
        "-f", "--is_file", action="store_true", help="默认是上传和下载文件夹"
    )

    parser.add_argument(
        "-r",
        "--remote",
        action="store",
        required=True,
        metavar="REMOTE_FOLDER_PATH",
        help="specify the remote file or folder path to download or upload.",
    )
    parser.add_argument(
        "-l",
        "--local",
        action="store",
        required=True,
        metavar="LOCAL_FOLDER_PATH",
        help="specify the local file or folder path to download or upload.",
    )
    args = parser.parse_args()

    print(args.__dict__)

    is_upload, is_file, remote, local = (
        args.is_up,
        args.is_file,
        args.remote,
        args.local,
    )

    # 上传
    if is_upload:
        print("上传...")
        upload_file_or_folder(
            local_file_folder=local, remote_folder=remote, is_file=is_file
        )
    else:
        print("下载...")
        down_file_or_folder(remote_path=remote, local_folder=local, is_file=is_file)


main()