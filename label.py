def get_emotion_label(excel_path):
    # 检查文件路径并加载相应的Excel文件
    if 'samm' in excel_path.lower():
        xl = pd.ExcelFile(excel_path, engine='openpyxl')
        df = xl.parse(xl.sheet_names[0], dtype={'Subject': str})
    else:
        xl = pd.ExcelFile(excel_path, engine='openpyxl')
        df = xl.parse(xl.sheet_names[0],dtype={'Subject': str})

    # 选择所需的列
    if 'casme^3' in excel_path:
        columns_to_select = ['Subject', 'Filename', 'Onset','emotion']
    else:
        columns_to_select = ['Subject', 'Filename', 'Estimated Emotion']

    df_selected = df[columns_to_select]
    if 'casme2' in excel_path.lower():
        # 拼接 'Subject' 和 'Filename' 生成 'Combined_Name' 列
        # df_selected['Combined_Name'] = 'sub' + df_selected['Subject'].astype(str) + '_' + df_selected['Filename'].astype(str)
        df_selected.loc[:, 'Combined_Name'] = 'sub' + df_selected['Subject'].astype(str) + '_' + df_selected[
            'Filename'].astype(str)
    elif 'casme^3' in excel_path:

        df_selected['Combined_Name'] = df_selected['Subject'] + '_' +df_selected['Subject'].astype(str) + '_' + df_selected['Filename'].astype(str)+ '_' + df_selected[
            'Onset'].astype(str)
    else:
        df_selected['Combined_Name'] = df_selected['Subject'].astype(str) + '_' + \
                                       df_selected['Filename'].astype(str)
    if 'casme^3' in excel_path:
        # 选择要返回的列
        result = df_selected[['Combined_Name', 'emotion']]

        # 将结果转换为字典，以便快速查询
        combined_data = {row['Combined_Name']: row['emotion'] for index, row in result.iterrows()}
    else:
        # 选择要返回的列
        result = df_selected[['Combined_Name', 'Estimated Emotion']]

        # 将结果转换为字典，以便快速查询
        combined_data = {row['Combined_Name']: row['Estimated Emotion'] for index, row in result.iterrows()}



    return combined_data


# 分类编码函数
def classify_emotion1(emotion):
    emotion = emotion.lower()
    # 定义情绪类别及编码
    if emotion == 'disgust':
        return 0
    elif emotion == 'happiness':
        return 2
    elif emotion == 'surprise':
        return 4
    elif emotion == 'repression':
        return 3
    elif emotion == 'others':
        return 1
def classify_emotion2(emotion):
    emotion = emotion.lower()
    # 定义情绪类别及编码
    if emotion == 'anger':
        return 0
    elif emotion == 'happiness':
        return 2
    elif emotion == 'surprise':
        return 4
    elif emotion == 'contempt':
        return 3
    elif emotion == 'other':
        return 1
def classify_emotion3(emotion):
    emotion = emotion.lower()
    # 定义情绪类别及编码
    if emotion == 'anger':
        return 0
    elif emotion == 'others':
        return 1
    elif emotion == 'happy':
        return 2
    elif emotion == 'fear':
        return 3
    elif emotion == 'surprise':
        return 4
    elif emotion == 'disgust':
        return 5
    elif emotion == 'sad':
        return 6
def classify_emotion4(emotion):
    emotion = emotion.lower()
    # 定义情绪类别及编码
    if emotion in['anger', 'fear',  'disgust', 'sad']:
        return 0
    elif emotion == 'others':
        return 1
    elif emotion == 'happy':
        return 2
    elif emotion == 'surprise':
        return 3
# def classify_emotion(emotion):
#     emotion = emotion.lower()
#     # 定义情绪类别及编码
#     if emotion == 'happy':
#         return 0
#     elif emotion == 'surprise':
#         return 1
#     elif emotion in['fear','sad','anger','disgust']:
#         return 2
#     elif emotion == 'others':
#         return 3
# def classify_emotion(emotion):
#     emotion = emotion.lower()
#     # 定义情绪类别及编码
#     if emotion == 'happiness':
#         return 0
#     elif emotion == 'surprise':
#         return 1
#     elif emotion in['fear','sadness','disgust','contempt','repression']:
#         return 2
#     elif emotion == 'anger':
#         return 3
#     elif emotion == 'others':
#         return 4
# 主函数示例
def process_dataset(data_name,cls,dataset, excel_path):
    subs = set()



    # 获取情绪数据
    combined_data = get_emotion_label(excel_path)

    # 遍历 dataset 并替换 labels
    processed_data = []
    for item in dataset:
        # 获取文件名（去掉路径和扩展名）
        filename = os.path.splitext(os.path.basename(item['path']))[0]
        # 分割文件名并去掉最后一个部分
        parts = filename.split('_')
        if len(parts) > 1:
            # filename = '_'.join(parts[:-1])  # 重新拼接，去掉最后一个部分
            filename = parts[1]+'_'+parts[1]+'_'+parts[2]+'_'+parts[3]


        # 重新加上文件扩展名
        new_filename = filename
        emotion = combined_data.get(new_filename)
        if emotion is None:
            print('error',filename)
        if data_name == 'casme2':
           emotion_label = classify_emotion1(emotion)
        elif data_name == 'samm':
            emotion_label = classify_emotion2(emotion)
        elif data_name == 'CAS(ME)^3'and cls==7:
            emotion_label = classify_emotion3(emotion)
        elif data_name == 'CAS(ME)^3' and cls==4:
            emotion_label = classify_emotion4(emotion)

        # 将 sub 加入集合中以防重复
        subs.add(item['sub'])

        # 更新 dataset 中的 labels
        processed_data.append({
            'path': item['path'],
            'sub': item['sub'],
            'labels': emotion_label  # 替换为查询到的情绪编码
        })

    return processed_data
