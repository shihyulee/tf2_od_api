import utils.DownloadMinio as DownloadMinio
import utils.CreateDataset as CreateDataset
import utils.CreateTrainFolder as CreateTrainFolder
import utils.RunTrain as RunTrain
import os

project_dir = "myproject"
output_folder = "tf_training"


def app_run(project_name, bucket, testRatio, num_steps, NN_architecture):

    try:
        # Get Images From minio
        Get_Images = DownloadMinio.download_images(
            project_dir, project_name, bucket)
        print(Get_Images)

        # Create Dataset
        make_dataset = CreateDataset.create_dataset(
            project_dir, project_name, testRatio)
        print(make_dataset)
        num_classes = make_dataset[2]

        # Create Train Folder
        CreateTrainFolder.Create_TrainFolder(
            project_dir, project_name, num_classes, num_steps, NN_architecture)

        # Train
        print("Start Training ...")
        project_folder = os.path.join(project_dir, project_name)
        output_folder_path = os.path.join(project_folder, output_folder)

        MODEL_DIR = os.path.join(output_folder_path, "train")
        config_output = os.path.join(output_folder_path, "pipeline.config")

        RunTrain.model_train_val(MODEL_DIR, config_output)
        print("Training Finished")
        msg = (1, "Finish")
    except Exception as e:
        msg = (0, e)
    return msg


if __name__ == '__main__':
    project_name = "project001"
    bucket = "project001"
    testRatio = 0.2
    num_steps = 2000
    NN_architecture = "resnet101"  # "resnet50",  "resnet101", "inception_resnet_v2"

    run_app = app_run(project_name, bucket, testRatio,
                      num_steps, NN_architecture)
