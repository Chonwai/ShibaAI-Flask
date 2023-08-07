import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from dotenv import load_dotenv
load_dotenv()
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

# First, let's load the language model we're going to use to control the agent.
# llm = OpenAI(openai_api_key=os.getenv(
#     "OPENAI_API_ACCESS_TOKEN"), temperature=0.7, model_name=os.getenv("OPENAI_MODEL_NAME"))
llm = ChatOpenAI(temperature=0.9, openai_api_key=os.getenv(
    "OPENAI_API_ACCESS_TOKEN"), model_name=os.getenv("OPENAI_MODEL_NAME"))

summary_form_data_prompt = PromptTemplate(
    input_variables=["query", "documents"],
    template="""
        Acts as a Data Engineer, could you help me to summarize the reference data to match \
        the description about '''{query}'''. You only extract the necessary data and process \
        them to match the demand description. \
        The unnecessary data you can remove them. You may have to do some calculation to \
        summarize the data if you meet the statistics scenario. \
        Use the Markdown format to output the result. \
        Summary Data: 'xxx' \
        Here is the reference data you need to summarize: \
        '''{documents}'''
        Summary Data:
    """,
)

generate_chart_prompt = PromptTemplate(
    input_variables=["query", "summarized_data"],
    template="""
        Acts as a Data Engineer, could you help me to implement the data analysis task on \
        '''{query}''' The output result I want you to make a chart by using highcharts.js and \
        give me an HTML JavaScript file!
        Use the format \
        <html> \
            <head> \
                <script src="https://code.highcharts.com/highcharts.js"></script> \
            </head> \
            <body> \
                <div id="container" style="width:100%; height:500px;"></div> \
            </body> \
        </html> \
        Here is the reference data you have to use: \
        {summarized_data}
        Output Result:
    """,
)


class ChainService:
    @ staticmethod
    def analysis(query, documents):
        chain1 = LLMChain(llm=llm, prompt=summary_form_data_prompt)
        summarized_data = chain1.run(query=query, documents=documents)
        print(summarized_data)
        print("----------------------")

        chain2 = LLMChain(llm=llm, prompt=generate_chart_prompt)
        chart = chain2.run(query=query, summarized_data=summarized_data)
        print(chart)
        print("----------------------")
        
        # Extract the HTML code from the text string

        # overall_chain = SimpleSequentialChain(
        #     chains=[chain1, chain2], verbose=True)

        # final_answer = overall_chain.run(
        #     "query={query}, documents={documents}".format(query=query, documents=documents))

        return chart
