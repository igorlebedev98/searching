from bs4 import BeautifulSoup

def extract_comments(html):
    soup = BeautifulSoup(html, 'html.parser')

    comments_data = []


    comment_blocks = soup.find_all('div', class_='feedback-item')

    for block in comment_blocks:
        # Извлекаем комментарий
        comment = block.find('div', class_='feedback-text').get_text(strip=True)
        
        # Извлекаем оценку
        rating = block.find('p', class_='feedback-rating').get_text(strip=True)
        
        # Извлекаем ID пользователя
        user_id = block['data-user-id']  # Предполагают, что ID находится в атрибуте

        # Сохраняем данные в словарь
        comments_data.append({
            'user_id': user_id,
            'rating': rating,
            'comment': comment
        })

    return comments_data

# Пример использования
html_content = """<html>... содержимое страницы ...</html>"""  # Здесь должен быть ваш HTML-код
comments = extract_comments(html_content)

for comment in comments:
    print(comment)
