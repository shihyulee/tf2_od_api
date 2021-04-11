import minio
import os

minio_conf = {
    'endpoint': ' ',
    'access_key': ' ',
    'secret_key': ' ',
    'secure': False
}

image_folder = 'images/'
label_folder = 'labels/'


def download_images(project_dir, project_name, bucket):
    client = minio.Minio(**minio_conf)
    if not client.bucket_exists(bucket):
        msg = "The bucket [{}] is not exist!".format(bucket)
        return (0, msg)

    #  建立下載目標之 list
    image_list = []
    objects = client.list_objects(bucket, prefix="images/")
    for obj in objects:
        image_list.append(obj.object_name)

    label_list = []
    objects = client.list_objects(bucket, prefix="labels/")
    for obj in objects:
        label_list.append(obj.object_name)

    download_list = image_list + label_list
    if len(download_list) == 0:
        msg = "Image is not found!"
        return (0, msg)

    # 建立專案目錄
    project_folder = os.path.join(project_dir, project_name)
    if not os.path.exists(project_folder):
        os.makedirs(project_folder)

    # 在專案目錄中建立下載資料夾
    download_data = "downloads"
    download_data_folder = os.path.join(project_folder, download_data)
    if not os.path.exists(download_data_folder):
        os.makedirs(download_data_folder)

    for i in download_list:
        try:
            filename = os.path.basename(i)
            save_file = os.path.join(download_data_folder, filename)
            client.fget_object(bucket, i, save_file)

        except Exception as err:
            print(err)
    return (1, "Images Downloaded Successful")


if __name__ == '__main__':
    bucket = "project001"
    project_dir = "myproject"
    project_name = "project001"

    download_images(project_dir, project_name, bucket)
