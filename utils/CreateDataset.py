import os
import glob
import random
import xml.etree.ElementTree as ET
import shutil


def create_dataset(project_dir, project_name, testRatio):
    # 下載影像與標記檔位置
    imageFolder = os.path.join(project_dir, project_name, 'downloads')

    # 建立 dataset - train 與 eval 資料夾
    imageTrain = os.path.join(project_dir, project_name, 'dataset', 'Train')
    imageEval = os.path.join(project_dir, project_name, 'dataset', 'eval')

    if not os.path.exists(imageTrain):
        os.makedirs(imageTrain)
    if not os.path.exists(imageEval):
        os.makedirs(imageEval)

    # 列出圖檔路徑
    fileList = []
    for file in os.listdir(imageFolder):
        filename, file_extension = os.path.splitext(file)
        file_extension = file_extension.lower()

        if(file_extension == ".jpg" or file_extension == ".jpeg" or file_extension == ".png" or file_extension == ".bmp"):
            fileList.append(os.path.join(imageFolder, file))

    testCount = int(len(fileList) * testRatio)
    trainCount = len(fileList) - testCount

    a = range(len(fileList))
    test_data = random.sample(a, testCount)
    train_data = [x for x in a if x not in test_data]

    train_count = 0
    for i in train_data:
        image_filename = fileList[i]
        save_image = os.path.join(imageTrain, os.path.basename(image_filename))
        filename, file_extension = os.path.splitext(image_filename)
        label_filename = filename + '.xml'
        save_label = os.path.join(imageTrain, os.path.basename(label_filename))

        if os.path.isfile(label_filename):
            shutil.copyfile(image_filename, save_image)
            shutil.copyfile(label_filename, save_label)
            train_count += 1

    test_count = 0
    for i in test_data:
        image_filename = fileList[i]
        save_image = os.path.join(imageEval, os.path.basename(image_filename))
        filename, file_extension = os.path.splitext(image_filename)
        label_filename = filename + '.xml'
        save_label = os.path.join(imageEval, os.path.basename(label_filename))

        if os.path.isfile(label_filename):
            shutil.copyfile(image_filename, save_image)
            shutil.copyfile(label_filename, save_label)
            test_count += 1

    data_dict = {"Train_Num": train_count, "Eval_Num": test_count}

    print("Create dataset: ", data_dict)

    # 建立 labels.names
    xml_list = []
    for xml_file in glob.glob(imageFolder + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = member[0].text
            xml_list.append(value)
    class_list = list(set(xml_list))
    class_num = len(class_list)

    obj_names = "labels.names"
    obj_names_path = os.path.join(
        project_dir, project_name, 'dataset', obj_names)

    with open(obj_names_path, 'w') as the_file:
        for className in class_list:
            the_file.write(className + "\n")
    print("Create [labels.names]")

    return (1, data_dict, class_num)


if __name__ == '__main__':
    project_dir = "myproject"
    project_name = "project001"

    testRatio = 0.2
    make_dataset = create_dataset(project_dir, project_name, testRatio)
