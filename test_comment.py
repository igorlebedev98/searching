import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from com_package.comment import Comment

from hamcrest import assert_that, is_in, less_than

class TestComment(unittest.TestCase):
    
    

    def setUp(self):
        self.comment = Comment(user_id="1234567890", text="Хороший товар, продовец отличный!", rating=5, timestamp=datetime.now())

    # Assertions
    def test_user_id_attribute(self):
        self.assertEqual(self.comment.user_id, "1234567890")  

    def test_text_attribute(self):
        self.assertEqual(self.comment.text, "Купите это по скидке!")  

    def test_rating_attribute(self):
        self.assertEqual(self.comment.rating, 5)  

    def test_repr_method(self):
        expected_repr = f"Comment(user_id={self.comment.user_id}, text={self.comment.text}, rating={self.comment.rating}, timestamp={self.comment.timestamp})"
        self.assertEqual(repr(self.comment), expected_repr) 

    def test_repr_method_with_different_values(self):
        new_comment = Comment(user_id="9876543210", text="Отличный продукт!", rating=4, timestamp=datetime.now())
        expected_repr = f"Comment(user_id={new_comment.user_id}, text={new_comment.text}, rating={new_comment.rating}, timestamp={new_comment.timestamp})"
        self.assertEqual(repr(new_comment), expected_repr)


    # assumption
    def test_rating_with_assumptions(self):
        if not (1 <= self.comment.rating <= 5):
            self.skipTest("Рейтинг не соответствует предполагаемым границам (1-5)")
        self.assertTrue(True) 

  
    def test_text_length_assumption(self):
        # Если длина текста больше 10000, пропускаем тест
        if len(self.comment.text) > 10000:
            self.skipTest("Текст превышает 10000 символов")
        self.assertTrue(True)  

    # matcher
    def test_rating_with_assumptions(self):
    # Проверяем, что рейтинг больше или равен 1
        assert_that(self.comment.rating, is_in(range(1, 6)), "Рейтинг должен быть в диапазоне от 1 до 5")
    
    # Проверяем, что рейтинг меньше или равен 5
        assert_that(self.comment.rating, less_than(6), "Рейтинг не должен превышать 5")

    # mocking
    @patch('com_package.comment.Comment.validate_user')
    def test_mock_validate_user(self, mock_validate_user):
        mock_validate_user.return_value = "1234567890"
        mock_comment = Comment(user_id="1234567890", text="Тест", rating=5)
        self.assertEqual(mock_comment.user_id, "1234567890")
        mock_validate_user.assert_called_once()

    @patch('com_package.comment.Comment.validate_timestamp')
    def test_mock_validate_timestamp(self, mock_validate_timestamp):
        mock_validate_timestamp.return_value = datetime.now()
        comment = Comment(user_id="1234567890", text="Тест", rating=5)
        self.assertIsInstance(comment.timestamp, datetime)
        mock_validate_timestamp.assert_called_once() 

    @patch('com_package.comment.Comment.is_bot', return_value=False)
    def test_mock_is_bot(self, mock_is_bot):
        self.assertFalse(self.comment.is_bot(["не бот"]))
        mock_is_bot.assert_called_once_with(["не бот"]) 

    # Параметризованные тестовые случаи
    def test_comment_with_different_user_ids_and_dates(self):
        test_cases = [
            ("1234567890", datetime.now(), True),                # Валидный user_id и текущая дата
            ("0987654321", datetime.now() - timedelta(days=1), True),  # Валидный user_id и дата на день назад
            ("1234567890", datetime(2005, 1, 1), True),          # Валидный user_id и дата в 2005
            ("9876543210", datetime(2000, 1, 1), False),         # Валидный user_id, но дата слишком старая
            ("123456789", datetime.now(), False),                # Невалидный user_id (не 10 цифр)
            ("1234567890", datetime(2025, 1, 1), True),          # Валидный user_id и дата в будущем
        ]

        for user_id, timestamp, expected in test_cases:
            with self.subTest(user_id=user_id, timestamp=timestamp):
                if expected:
                    comment = Comment(user_id=user_id, text="Я бот!", rating=3, timestamp=timestamp)
                    self.assertEqual(comment.user_id, user_id)  # Проверка корректного user_id
                else:
                    with self.assertRaises(ValueError):
                        Comment(user_id=user_id, text="Я бот!", rating=2, timestamp=timestamp)


# Запуск тестов
if __name__ == "__main__":
    unittest.main()
