import os
import pandas as pd


def write_filenames_to_excel(image_folder_path, excel_path):
    # 获取图片文件夹中的所有文件名
    filenames = os.listdir(image_folder_path)
    filenames.sort()
    origin_names = []
    labels = []
    for name in filenames:
        origin_names.append(int(name[:-4].split('_')[0]))
        labels.append(name[:-4].split('_')[1])
    # 创建DataFrame对象
    df = pd.DataFrame({'原始文件名': origin_names, '原始标签': labels, '原始标签（需修改）': labels})
    df['原始文件名'] = df['原始文件名'].astype(int)  # 将原始文件名列转换为整数类型
    df = df.sort_values(by='原始文件名')
    # 将DataFrame写入Excel表格
    df.to_excel(excel_path, index=False)

    # df_2.to_excel(excel_path, index=False)
    print("File names written to Excel successfully.")


# 示例用法
image_folder_path = r'D:\wise_transportation\data\146-9-auto-label'
excel_path = r'D:\wise_transportation\data\146-9-auto-label\label.xlsx'

write_filenames_to_excel(image_folder_path, excel_path)