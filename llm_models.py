from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
import os

def getDeepSeek():
    return ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        api_key=os.getenv("DEEPSEEK_API_KEY")
    )

def getOpenAI():
    return ChatOpenAI(
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
