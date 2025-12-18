import pytest
from pytest_bdd import scenarios, when, then
from com_package.bot_or_not import bot_or_not

scenarios("mock_bot.feature")

@when('I call bot_or_not with "{text}"')
def call_bot_function(text):
    """Вызываем функцию с заданным текстом"""
    # Результат сохраняем в контексте теста pytest
    pytest.result = bot_or_not(text)

@then("the result should be a boolean")
def verify_boolean():
    """Проверяем что результат - boolean"""
    result = pytest.result
    
    print(f"Result: {result}")
    print(f"Type: {type(result)}")
    
    # Проверяем что это bool
    assert isinstance(result, bool), f"Expected bool, got {type(result)}"
    
    # Проверяем что это именно True или False
    assert result in [True, False], f"Expected True or False, got {result}"