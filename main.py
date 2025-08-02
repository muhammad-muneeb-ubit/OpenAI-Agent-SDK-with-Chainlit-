from agents import Agent, Runner, OpenAIChatCompletionsModel, RunConfig, AsyncOpenAI
import os
from dotenv import load_dotenv
import chainlit as cl  
import asyncio
load_dotenv()

gemini_Key = os.getenv("GEMINI_API_KEY")
if not gemini_Key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

external_client = AsyncOpenAI(
    api_key=gemini_Key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=external_client,
)

config = RunConfig( 
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)
agent = Agent(
    name="Helping Agent in the field of Frontend web Development",
    instructions="An agent that helps with various tasks using Gemini API in Frontend web Development.",
)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(
        content="Welcome to the Frontend Web Development Assistant! How can I help you today?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})
    result = await Runner.run(
        agent,
        input=history,
        run_config=config,
    )
    history.append({"role": "user", "content": result.final_output})
    cl.user_session.set("history", history)
    await cl.Message(content=result.final_output).send()



# print("result ---->" ,result.final_output)
