import json
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from main import main 

## В файле работа приложения в части UI

class MyWidget(BoxLayout):
    def __init__(self, **kwargs):
        super(MyWidget, self).__init__(**kwargs)

        self.text_input = TextInput(hint_text='Введите текст здесь', multiline=False)
        self.add_widget(self.text_input)

        self.button = Button(text='Проанализировать')
        self.button.bind(on_press=self.analyze)
        self.add_widget(self.button)

        self.result_label = Label(size_hint_y=None, height=44)
        self.add_widget(self.result_label)

    def analyze(self, instance):
        input_text = self.text_input.text
        self.process_input(input_text)

    def process_input(self, text):
        result = main(text) 
        self.display_result(result)

    def display_result(self, result):
        try:
            data = json.loads(result)  
            mark = data.get('mark', 'Нет данных')
            procent = data.get('procent', 'Нет данных')
            self.result_label.text = f'Оценка ИИ для товара: {mark}\nПроцент найденных ботов: {procent}'
        except json.JSONDecodeError:
            self.result_label.text = 'Ошибка при обработке результата.'

class MyApp(App):
    def build(self):
        return MyWidget()

if __name__ == '__main__':
    MyApp().run()
