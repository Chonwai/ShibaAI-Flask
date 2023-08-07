import os
from langchain.agents import load_tools, Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.utilities import GoogleSerperAPIWrapper
from dotenv import load_dotenv
load_dotenv()
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

# First, let's load the language model we're going to use to control the agent.
llm = OpenAI(openai_api_key=os.getenv(
    "OPENAI_API_ACCESS_TOKEN"), temperature=0.7, model_name=os.getenv("OPENAI_MODEL_NAME"))

# search = SerpAPIWrapper(verbose=True, api_key=os.getenv("SERPAPI_API_KEY"), engine="google", num_results=10)
search = GoogleSerperAPIWrapper(verbose=True, num_results=10)
llm_math_chain = LLMMathChain(llm=llm, verbose=True)

# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
# tools = load_tools(["serpapi", "llm-math"], llm=llm)
tools = [
    Tool(
        name="Search First",
        func=search.run,
        description="useful for when you need to find more related information about the topic or input on the internet"
    ),
    Tool(
        name="Translation",
        func=llm,
        description="useful for when you need to translate the script"
    ),
    Tool(
        name="Generation",
        func=llm,
        description="useful for when you need to generate some content"
    ),
    Tool(
        name="Finalization",
        func=llm,
        description="useful for when you need to finalize the script"
    )
]

docai_tools = [
    Tool(
        name="Analysis",
        func=llm,
        description="useful for when you need to analyze the task first"
    ),
    Tool(
        name="Extraction",
        func=llm,
        description="useful for when you need to extract the reference data from the task"
    ),
    Tool(
        name="Summarization",
        func=llm,
        description="useful for when you need to summarize and remove the unnecessary data"
    ),
    Tool(
        name="Generation",
        func=llm,
        description="useful for when you need to generate some highcharts.js content, such as charts, tables or text"
    ),
    Tool(
        name="Finalization",
        func=llm,
        description="useful for when you need to finalize the result to HTML code"
    )
]

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
# agent = initialize_agent(
#     tools, llm, agent=AgentType.conversational_react_description, verbose=True)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

docai_agent = initialize_agent(
    docai_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


class AgentService:
    @ staticmethod
    def run(style, length, description, region, hashtagCount, tone, language):
        # You are now an ${data.language} ${data.style} social media script writer and your tone style is ${data.tone}! The output must follow the json format (the newline symbol "\n" must be replace to "\\n"!) {"content": ""}! Please extend and generate a ${data.length} size script and include emoji with ${data.region} style script with ${data.language} and give ${data.hashtagCount} related topic popular hashtags based on the given reference description: ${data.description}.

        # Now let's test it out!
        # output = agent.run(
        #     f"Acts as a English Instagram social media influencer. Your script tone style is Casual 🤣 and including some emoji in the script. Could you please write a long social media script with Hong Kong 🇭🇰 style and give 20 related popular hashtags. The generate script topic is about: '''{input}'''")
        print(style, length, description, region, hashtagCount, tone, language)
        output = agent.run(
            'Acts as a {language} {style} social media script writer and your tone style is {tone}! The output must follow the json format (the newline symbol "\n" must be replace to "\\n"!) ```{{"content": ""}}```! Please extend and generate a {length} size script and include emoji with {region} style script with {language} and give {hashtagCount} related topic popular hashtags based on the given reference description: ```{description}```.'.format(style=style, length=length, description=description, region=region, hashtagCount=hashtagCount, tone=tone, language=language))
        print(output)
        return output

    @ staticmethod
    def analyze(query='幫我總結一下哪一個部門最多人請假？'):
        print(query)
        output = docai_agent.run(
            'Acts as a Data Engineer, could you help me to implement the data analysis task on ```{query}```. The output result I want you to make a chart by using highcharts.js and only give me an HTML JavaScript file! The final result I need a HTML code. Here is the data need to analysis for your reference: ```data = [{{ "employee_id": "307", "employee_name": "員 工 姓 名 : 林 一 二", "type_of_leave": {{ "no_timeout": false, "sick_leave": true, "annual_leave": false, "cancel_leave": false, "official_leave": false, "personal_leave": false }}, "date_of_filling": "2022/1/21", "employee_position": "员 工", "reason_of_absence": "看 病", "working_department": {{ "309": false, "bga": false, "cpg": false, "cun": false, "fty": false, "glp": false, "hzm": false, "mgm": false, "off": false, "sml": true, "tft": false, "tpa": false, "yao": false, "tsb5": false, "cpg20": false, "other": false, "tsb28": false, "other_department": null }}, "duration_of_absence": {{ "date_of_absence_to": "2022/1/22", "time_of_absence_to": null, "date_of_absence_from": "2022/1/21", "time_of_absence_from": null }}, "administrative_approval": {{ "agree": true, "disagree": false }}, {{ "employee_id": "137", "employee_name": "郭大文", "type_of_leave": {{ "no_timeout": false, "sick_leave": false, "annual_leave": false, "cancel_leave": true, "official_leave": false, "personal_leave": false }}, "date_of_filling": "2022/6/18", "employee_position": "工程技術人員", "reason_of_absence": "魏 得間", "working_department": {{ "309": false, "bga": false, "cpg": false, "cun": false, "fty": false, "glp": false, "hzm": false, "mgm": false, "off": false, "sml": true, "tft": false, "tpa": false, "yao": false, "tsb5": false, "cpg20": false, "other": false, "tsb28": false, "other_department": null }}, "duration_of_absence": {{ "date_of_absence_to": "2022/9/7", "time_of_absence_to": null, "date_of_absence_from": "2022/9/6", "time_of_absence_from": null }}, "administrative_approval": {{ "agree": true, "disagree": false }}, {{ "employee_id": "281", "employee_name": "李清照", "type_of_leave": {{ "no_timeout": false, "sick_leave": false, "annual_leave": false, "cancel_leave": false, "official_leave": false, "personal_leave": true }}, "date_of_filling": "2021/4/20", "employee_position": "员工", "reason_of_absence": "灵感来了,我需要回家写词。", "working_department": {{ "309": false, "bga": false, "cpg": false, "cun": true, "fty": false, "glp": false, "hzm": false, "mgm": false, "off": false, "sml": false, "tft": false, "tpa": false, "yao": false, "tsb5": false, "cpg20": false, "other": false, "tsb28": false, "other_department": null }}, "duration_of_absence": {{ "date_of_absence_to": "2021/4/22", "time_of_absence_to": "08:30", "date_of_absence_from": "2021/4/20", "time_of_absence_from": "14:00" }}, "administrative_approval": {{ "agree": true, "disagree": false }}, {{ "employee_id": "137", "employee_name": "郭 大 文", "type_of_leave": {{ "no_timeout": false, "sick_leave": false, "annual_leave": false, "cancel_leave": true, "official_leave": false, "personal_leave": false }}, "date_of_filling": "2022/6/18", "employee_position": "工程技術人員", "reason_of_absence": "得 間", "working_department": {{ "309": false, "bga": false, "cpg": false, "cun": false, "fty": false, "glp": false, "hzm": false, "mgm": false, "off": false, "sml": true, "tft": false, "tpa": false, "yao": false, "tsb5": false, "cpg20": false, "other": false, "tsb28": false, "other_department": null }}, "duration_of_absence": {{ "date_of_absence_to": "2022/9/7", "time_of_absence_to": null, "date_of_absence_from": "2022/9/6", "time_of_absence_from": null }}, "administrative_approval": {{ "agree": true, "disagree": false }}, {{ "employee_id": "137", "employee_name": "林小文", "type_of_leave": {{ "no_timeout": false, "sick_leave": true, "annual_leave": false, "cancel_leave": false, "official_leave": false, "personal_leave": false }}, "date_of_filling": "2022/1/14", "employee_position": "员工", "reason_of_absence": "身体不适", "working_department": {{ "309": false, "bga": false, "cpg": false, "cun": false, "fty": false, "glp": true, "hzm": false, "mgm": false, "off": false, "sml": false, "tft": false, "tpa": false, "yao": true, "tsb5": false, "cpg20": false, "other": false, "tsb28": false, "other_department": null }}, "duration_of_absence": {{ "date_of_absence_to": "2022/1/15", "time_of_absence_to": "08:30", "date_of_absence_from": "2022/1/14", "time_of_absence_from": "08:30" }}, "administrative_approval": {{ "agree": true, "disagree": false }}, {{ "employee_id": "755431", "employee_name": "王 小 明", "type_of_leave": {{ "no_timeout": false, "sick_leave": true, "annual_leave": false, "cancel_leave": false, "official_leave": false, "personal_leave": false }}, "date_of_filling": "2022/6/12", "employee_position": null, "reason_of_absence": "腸 胃 不 適", "working_department": {{ "309": false, "bga": false, "cpg": true, "cun": false, "fty": false, "glp": false, "hzm": false, "mgm": false, "off": false, "sml": false, "tft": false, "tpa": false, "yao": false, "tsb5": false, "cpg20": false, "other": false, "tsb28": false, "other_department": null }}, "duration_of_absence": {{ "date_of_absence_to": "2022/6/13", "time_of_absence_to": null, "date_of_absence_from": "2022/6/12", "time_of_absence_from": null }}, "administrative_approval": {{ "agree": true, "disagree": false }}, {{ "employee_id": "956", "employee_name": "44.", "type_of_leave": {{ "no_timeout": false, "sick_leave": false, "annual_leave": false, "cancel_leave": true, "official_leave": false, "personal_leave": false }}, "date_of_filling": ":2022-11-4", "employee_position": "店 务 员", "reason_of_absence": "协 助 同 事 调 休", "working_department": {{ "309": false, "bga": false, "cpg": false, "cun": null, "fty": false, "glp": null, "hzm": false, "mgm": false, "off": true, "sml": false, "tft": null, "tpa": false, "yao": null, "tsb5": null, "cpg20": false, "other": false, "tsb28": false, "other_department": null }}, "duration_of_absence": {{ "date_of_absence_to": "2022-11-2.3 号", "time_of_absence_to": null, "date_of_absence_from": ")2022-11-4", "time_of_absence_from": null }}, "administrative_approval": {{ "agree": true, "disagree": false }}, {{ "employee_id": "768", "employee_name": "孔子明", "type_of_leave": {{ "no_timeout": false, "sick_leave": false, "annual_leave": false, "cancel_leave": false, "official_leave": false, "personal_leave": true }}, "date_of_filling": ":2022/6/18", "employee_position": "檔案事員", "reason_of_absence": "請假一天\n22", "working_department": {{ "309": false, "bga": false, "cpg": false, "cun": false, "fty": false, "glp": false, "hzm": false, "mgm": false, "off": false, "sml": false, "tft": false, "tpa": false, "yao": false, "tsb5": false, "cpg20": false, "other": true, "tsb28": false, "other_department": "MO" }}, "duration_of_absence": {{ "date_of_absence_to": "2022/8/10", "time_of_absence_to": null, "date_of_absence_from": "2022/8/9", "time_of_absence_from": null }}, "administrative_approval": {{ "agree": true, "disagree": false }}]```'.format(query=query))
        print(output)
        return output

    @ staticmethod
    def about():
        return 'About'
