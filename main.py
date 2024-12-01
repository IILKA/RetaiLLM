# from flask import Flask, request, jsonify

# app = Flask(__name__)

# @app.route('/chat', methods=['GET', 'POST'])
# def chat():
#     if request.method == 'POST':
#         data = request.json
#         message = data.get('message', "No message found")
#         print("message:", message)
#         return jsonify({"response": "I have received your message"})
#     else:
#         return jsonify({"response": "Please send a POST request"})

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8080, debug=True)


'''
import yaml
from retaillm import RetaiLLM
from flask import Flask, request, jsonify

def main():
    config = yaml.safe_load(open("./config.yaml"))
    agent_config = config["Agent"]

    while True: 
        user_input = input("enter a sentence: ")    
        retaillm = RetaiLLM(
            api_key = agent_config["api_key"],
            base_url = agent_config["base_url"],
            model_id = agent_config["model_id"],
            debug = agent_config["debug"]
        )
        response = retaillm.chat(user_input)
        print("response", response)
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port = 8080, debug=True)

main()
    
'''

import yaml
from flask import Flask, request, jsonify
from retaillm import RetaiLLM

app = Flask(__name__)

# Load configuration once at startup
with open("./config.yaml") as f:
    config = yaml.safe_load(f)

agent_config = config["Agent"]

# Initialize RetaiLLM once
retaillm = RetaiLLM(
    api_key=agent_config["api_key"],
    base_url=agent_config["base_url"],
    model_id=agent_config["model_id"],
    debug=agent_config["debug"]
)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'user_input' not in data:
        return jsonify({"error": "Missing 'user_input' in request"}), 400
    
    user_input = data['user_input']
    response = retaillm.chat(user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)

