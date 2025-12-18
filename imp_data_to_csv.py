import csv

def imp_to_csv(data, output_file):
    if not data:
        print("Нет данных для записи.")
        return

    headers = data[0].keys()
    headers = list(headers) + ['bot']

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
    
        writer.writeheader()
        for row in data:
            row['bot'] = '0'  
            writer.writerow(row)

comments_data = [
    {'user_id': '123', 'rating': '5', 'comment': 'Отлично!'},
    {'user_id': '456', 'rating': '4', 'comment': 'Хорошо, но есть недостатки.'},
    {'user_id': '789', 'rating': '3', 'comment': 'Средний товар.'}
]

dict_to_csv(comments_data, 'comments_data_with_bot.csv')
