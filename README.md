# emotion_vis_server
emotion visualization server
Tested with python-3.8.5

### Installation
- Clone the repository
`$git clone https://github.com/texsmv/emotion_vis_server`
- Enter to the repository dir
`$cd emotion_vis_server`
- Create an environment
`$python3 -m venv eviz`
- Activate the environment
`$source eviz/bin/activate `
- Install the server requirements
`$pip install -r requirements.txt`

### Start the flask server
- Be sure to activate the environment
`$source eviz/bin/activate `
- Run the local server
`$python app.py`

### Start the http server
- Go to the ui directory
`$cd ui`
- Run an http server
`$python -m http.server`
- Open the adress http://0.0.0.0:8000/ in a web browser. And that's all!

