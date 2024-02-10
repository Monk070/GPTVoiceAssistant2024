# ChatGPT Voice Chat 
Uses OpenAI Whisper to transcribe audio from spoken input. Working as of February 10th 2024 

## 🛠️Setup 

### 1. API Key: 

Replace `client = OpenAI()' with 'client = OpenAI(api_key="YOUR_API_KEY")` 

or

Go to Control Panel > Edit the system environment variable, and add an OPENAI_API_KEY

### 2. Dependencies 

```bash
pip install numpy pyaudio collections wave openai pygame
```

### 3. Run the Script: 

Execute the main script
```bash
python GPTVoiceAssistant2024.py
```

## 🎙 How to use: 

Talk into your microphone and speak your heart out with the bot. 

## 🤝 Contribute

Feel free to fork, improve, and submit pull requests. If you're considering significant changes or additions, please start by opening an issue.

## 💖 Acknowledgements

Huge shoutout to:
- [KoljaB On GitHub](https://github.com/KoljaB/AIVoiceChat/) for being amazing at what he does and inspiring me to create this
- [OpenAI](https://www.openai.com/) for pioneering with the GPT-4 model.
