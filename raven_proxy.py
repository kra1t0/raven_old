import re
import subprocess
import os
import json
import threading
import uuid
from flask import Flask, request, jsonify
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

app = Flask(__name__)

# Constants
BASE_DIR = "/opt/raven/users"
MEMORY_LIMIT = 10
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL_NAME = "llama3-8b-8192"

# Initialize Groq Chat client
groq_chat = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)

SYSTEM_PROMPT = """
You are Raven, a cybersecurity-focused AI assistant. Janindu is the owner of this system.
When you need to run a system command, wrap it in (COMMAND) and (COMMAND), like this:
(COMMAND) ls -la (COMMAND)
Do not add explanations inside the command block. I will run it for you and return the output immediately, then forward the output back to you for analysis.
"""

# Helpers for memory paths
def get_user_dir(user_id):
    path = os.path.join(BASE_DIR, str(user_id))
    os.makedirs(path, exist_ok=True)
    return path

def get_memory_path(user_id):
    return os.path.join(get_user_dir(user_id), "memory.json")

def get_archive_path(user_id):
    return os.path.join(get_user_dir(user_id), "archive.json")

# Memory management
def load_memory(user_id):
    path = get_memory_path(user_id)
    if os.path.exists(path):
        return json.load(open(path))
    return []


def save_memory(user_id, history):
    with open(get_memory_path(user_id), 'w') as f:
        json.dump(history, f, indent=2)


def archive_old_messages(user_id, history):
    if len(history) <= MEMORY_LIMIT:
        return history
    old, new = history[:-MEMORY_LIMIT], history[-MEMORY_LIMIT:]
    archive = []
    arch_path = get_archive_path(user_id)
    if os.path.exists(arch_path):
        archive = json.load(open(arch_path))
    archive.extend(old)
    with open(arch_path, 'w') as f:
        json.dump(archive, f, indent=2)
    return new

# Command extraction
def extract_commands(text):
    # Find all blocks wrapped by (COMMAND)...(COMMAND)
    return [cmd.strip() for cmd in re.findall(r"\(COMMAND\)(.*?)\(COMMAND\)", text, re.DOTALL)]

# Command execution
def run_command(cmd):
    try:
        completed = subprocess.run(
            cmd, shell=True, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=60
        )
        return completed.stdout.strip() or completed.stderr.strip()
    except Exception as e:
        return f"Error executing `{cmd}`: {e}"

# LLM interaction
def call_groq(history, user_input):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}")
    ])
    memory = ConversationBufferWindowMemory(
        k=MEMORY_LIMIT,
        memory_key="chat_history",
        return_messages=True
    )
    memory.chat_memory.messages = history.copy()
    chain = LLMChain(
        llm=groq_chat,
        prompt=prompt,
        memory=memory,
        verbose=False
    )
    return chain.predict(human_input=user_input)

# Chat endpoint
@app.route('/chat', methods=['POST'])
def handle_chat():
    try:
        data = request.get_json()
        user_input = data.get('input', '').strip()
        user_id = data.get('user_id', 'default')
        req_id = str(uuid.uuid4())

        if not user_input:
            return jsonify({'error': 'No input provided'}), 400

        # Load and prune history
        history = load_memory(user_id)
        history = archive_old_messages(user_id, history)

        # 1) Get Raven's suggestion
        suggestion = call_groq(history, user_input)

        # Update memory with suggestion
        history.append({'role': 'assistant', 'content': suggestion})

        # 2) Extract and run commands asynchronously
        cmds = extract_commands(suggestion)
        outputs = {}
        if cmds:
            threads = []
            for cmd in cmds:
                t = threading.Thread(
                    target=lambda c: outputs.setdefault(c, run_command(c)),
                    args=(cmd,)
                )
                t.start()
                threads.append(t)
            for t in threads:
                t.join()

        # 3) Build analysis input
        analysis_input = suggestion
        if outputs:
            # Append raw outputs to memory
            for cmd, out in outputs.items():
                history.append({'role': 'assistant', 'content': f"Output of `{cmd}`:\n{out}"})
            # Combine outputs for LLM
            combined = "\n\n".join(
                f"Output of `{c}`:\n{outputs[c]}" for c in cmds
            )
            analysis = call_groq(history, combined)
            history.append({'role': 'assistant', 'content': analysis})
        else:
            analysis = ''

        # Save updated history
        save_memory(user_id, history)

        # Return structured response
        return jsonify({
            'request_id': req_id,
            'suggestion': suggestion,
            'outputs': outputs,
            'analysis': analysis
        })
    except Exception as e:
        print(f"Error in /chat [{req_id}]: {e}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)