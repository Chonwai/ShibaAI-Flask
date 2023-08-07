from langchain import BaseChain

class LLMChartChain(BaseChain):
    def __init__(self):
        super().__init__()

    def prompt(self):
        return "What is your name?"

    def process(self, response):
        self.name = response

    def prompt2(self):
        return f"Hello {self.name}, how are you today?"

    def process2(self, response):
        self.mood = response

    def run(self):
        self.prompt()
        self.process(self.get_input())
        self.prompt2()
        self.process2(self.get_input())
        print(f"{self.name} is {self.mood} today.")
