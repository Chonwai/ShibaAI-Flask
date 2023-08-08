from flask import Flask, request, jsonify
from dotenv import load_dotenv
from api.services.agent.agent import AgentService
from api.services.chain.chain import ChainService
from api.services.qabot.qabot import QABotService
from flask_cors import CORS
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
        # requestData = request.get_json()
        # query = requestData['query']
        # metadata = requestData['metadata'] or {}
        query = request.args.get('query')
        metadata = request.args.get('metadata')
        res = QABotService.qaEvent(query, metadata)
        return jsonify({'status': True, 'content': res})
    except Exception as e:
        return jsonify({'status': False, 'message': str(e)})
