import csv
from pathlib import Path
from datetime import datetime
import pytest
from pytest_bdd import scenarios, given, when, then, parsers
from unittest import mock

from com_package.comment import Comment, save_if_bot 


scenarios("save_bot_comments.feature")


@given(parsers.parse('csv file path "{filename}"'))
def csv_file_path(tmp_path, filename):
    path = tmp_path / filename
    if path.exists():
        path.unlink()
    return str(path)


@given(parsers.parse('a user id "{user_id}"'))
def given_user_id(user_id):
    return user_id

@given(parsers.parse('a text "{text}"'))
def given_text(text):
    return text


@given(parsers.parse('a rating {rating:d}'))
def given_rating(rating):
    return rating

@given(parsers.parse('a user id "{user_id}"'))
def user_id(user_id):
    return user_id

@when("a Comment is created")
def create_comment(user_id, text, rating):
    return Comment(user_id=given_user_id, text=given_text, rating=given_rating)


@when("save_if_bot is called with the csv file")
def call_save_if_bot(monkeypatch, create_comment, csv_file_path):
    # Детектор: мокаем поведение для детерминированных тестов
    def fake_bot_or_not(text):
        lowered = text.lower()
        if "limited offer" in lowered or "buy now" in lowered:
            return True
        return False

    # Подменяем функцию в том модуле, где она импортируется
    monkeypatch.setattr("com_package.bot_or_not", fake_bot_or_not)

    result = save_if_bot(create_comment, csv_file_path)
    return {"result": result, "csv_path": csv_file_path, "comment": create_comment}


@then("the function returns true")
def function_returns_true(call_save_if_bot):
    assert call_save_if_bot["result"] is True


@then("the function returns false")
def function_returns_false(call_save_if_bot):
    assert call_save_if_bot["result"] is False


@then("the comment exists as a row in the CSV file")
def comment_exists_in_csv(csv_file_path, create_comment):
    path = Path(csv_file_path)
    assert path.exists()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    matches = [
        r for r in rows
        if r.get("user_id") == create_comment.user_id
        and r.get("text") == create_comment.text
        and r.get("rating") == str(create_comment.rating)
    ]
    assert matches, f"No matching row found in {path}"



@then("the comment does not exist as a row in the CSV file")
def comment_not_in_csv(csv_file_path, create_comment):
    path = Path(csv_file_path)
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    target = create_comment
    matches = [
        r for r in rows
        if r.get("user_id") == target.user_id
        and r.get("text") == target.text
        and r.get("rating") == str(target.rating)
    ]
    assert not matches, f"Found unexpected matching row in {path}"

