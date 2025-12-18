import pandas as pd
from googletrans import Translator

def data_translation (input_file, output_file):
    df = pd.read_csv(input_file)

    translator = Translator()

    def translate_text(text):
        try:
            translated = translator.translate(text, dest='ru')
            return translated.text
        except Exception as e:
            print(f'Ошибка при переводе: {e}')
            return text 

    df['Translated Tweet'] = df['Tweet'].apply(translate_text)

    translated_df = df[['Translated Tweet', 'Bot Label']]
    translated_df.to_csv(output_file, index=False)

data_translation('bot_detection_data.csv', 'translate_dataset.csv')


