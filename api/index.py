from flask import Flask, request, Response, jsonify, stream_with_context
from dotenv import load_dotenv
from api.services.agent.agent import AgentService
from api.services.chain.chain import ChainService
from api.services.qabot.qabot import QABotService
# from services.qabot.qabot import QABotService
from flask_cors import CORS
import time
load_dotenv()

app = Flask(__name__)
CORS(app, resources={
     r"/*": {"origins": ["*", "https://doc-ai-frontend-oqag5r4lf-chonwai.vercel.app/", "https://doc-ai-frontend.vercel.app/"]}})


@app.route('/')
def home():
    return 'Hello, World!'


@app.route('/run')
def run():
    style = request.args.get('style')
    length = request.args.get('length')
    description = request.args.get('description')
    region = request.args.get('region')
    hashtagCount = request.args.get('hashtagCount')
    tone = request.args.get('tone')
    language = request.args.get('language')
    res = AgentService.run(style, length, description,
                           region, hashtagCount, tone, language)
    return res


@app.route('/analyze')
def analyze():
    query = request.args.get('query')
    documents = request.args.get('documents')
    res = ChainService.analysis(query, documents)
    return res


@app.route('/about')
def about():
    AgentService.about()
    return 'About'


@app.route('/events/qa')
def eventsQA():
    try:
        query = request.args.get('query')
        res = QABotService.qaEvent(query)
        return jsonify({'status': True, 'content': res})
    except Exception as e:
        return jsonify({'status': False, 'message': str(e)})


@app.route('/events/qa/steaming')
def eventsQAStreaming():
    try:
        query = request.args.get('query')

    #     def generate():
    #         for ratio in query:
    #             yield str(ratio)
    #             print("ratio:", ratio)
    #             time.sleep(0.5)
    #     return Response(stream_with_context(generate()), mimetype='text/event-stream')
        return Response(QABotService.qaEventStreaming(query), mimetype="text/event-stream")
        # def event_stream():
        #     for line in QABotService.qaEvent(query):
        #         print(line)
        #         text = line.choices[0].delta.get('content', '')
        #         if len(text):
        #             yield text
        #         # print("Response: {}".format(res))
        #         # yield f"data: {res}\n\n"
        #         # yield jsonify({'status': True, 'content': res})
    # return Response(stream_with_context(event_stream()), mimetype='text/event-stream')
    except Exception as e:
        print("Error: {}".format(e))
        return jsonify({'status': False, 'message': str(e)})


# if __name__ == '__main__':
#     app.run(debug=True)
