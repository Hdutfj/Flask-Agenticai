from flask import Flask, render_template, request
import asyncio
from mode import outview

app = Flask(__name__)

# Store response history as list of dicts {user:..., bot:...}
history = []

@app.route('/', methods=['GET', 'POST'])
def index():
    response = ""
    if request.method == 'POST':
        user_input = request.form.get('input_text', '')
        action = request.form.get('action')

        if action == "ask_agent" and user_input.strip():
            try:
                response = asyncio.run(outview(user_input))
                history.append({'user': user_input, 'bot': response})
            except Exception as e:
                response = f"<strong>Error:</strong> {str(e)}"
    summary="/n".join([f"{m['user'].capitalize()}:{m['bot'][:50].strip()}...." for m in history])
    return render_template("chatter.html", response=response, history=history, summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
