import pandas as pd
import os


def rename_images_from_excel(excel_path, image_folder_path):
    # 读取Excel表格
    df = pd.read_excel(excel_path)

    # 遍历每一行
    for index, row in df.iterrows():
        original_filename = str(row['原始文件名']) + '_' + str(row['原始标签'])
        new_filename = str(row['原始文件名']) + '_' + str(row['label'])
        # 构建原始文件路径和调整后的文件路径
        original_path = os.path.join(image_folder_path, original_filename + '.jpg')
        new_path = os.path.join(image_folder_path, new_filename + '.jpg')

        # 重命名文件
        try:
            os.rename(original_path, new_path)
            print(f"Renamed image: {original_filename} -> {new_filename}")
        except:
            print("Deleted")

    print("Rename images completed.")


# 示例用法
excel_path = r'D:\wise_transportation\data\label.xlsx'
image_folder_path = r'D:\wise_transportation\data\146-9-auto-label'

rename_images_from_excel(excel_path, image_folder_path)
