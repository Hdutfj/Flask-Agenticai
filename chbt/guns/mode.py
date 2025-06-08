from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, InputGuardrail, GuardrailFunctionOutput, set_tracing_disabled, function_tool
from dotenv import load_dotenv
import os
from fpdf import FPDF
import uuid
from pydantic import BaseModel
from typing import Dict
import asyncio

class EnvInt(BaseModel):
    inputadd:str
    intent:str

# Data model for Guardrail output
class EnviromentOutput(BaseModel):
    is_Enviroment: bool
    reasoning: str
    intent: str

# Load API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
set_tracing_disabled(True)

# OpenAI Provider Config
provider = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
)

# Guardrail Agent
guardrailAgent = Agent(
    name="GuardrailAgent",
    instructions="You should check whether the input is of the correct type or not. If not, give an error.",
    output_type=EnviromentOutput,
    model=OpenAIChatCompletionsModel(model="gemini-1.5-flash", openai_client=provider)
)

# Guardrail Function
async def EnviromentGuardrail(ctx, agent, input_type):
    result = await Runner.run(guardrailAgent, input_type, context=ctx)
    final_output = result.final_output_as(EnviromentOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=False
    )

@function_tool
async def textshow(input: EnvInt):
    inputadd = input.inputadd
    intent = input.intent

    if intent == "generate_email":
        return f"Your email has been typed correctly: {inputadd}"

    elif intent == "convert_to_pdf":
        filename = f"{uuid.uuid4().hex}.pdf"
        filepath = f"static/pdfs/{filename}"

        os.makedirs("static/pdfs", exist_ok=True)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, inputadd)
        pdf.output(filepath)

        return f'Your PDF has been generated successfully. <a href="http://localhost:5000/static/pdfs/{filename}" target="_blank">Click here to view PDF</a>'
    
    else:
        return "Error: Unknown intent"


NatEnv_Agent = Agent(
    name="NatEnv Agent",
    instructions="""
    Answer all questions related to the natural environment, including:
    - Air (atmosphere)
    - Water (oceans, rivers, lakes)
    - Land (soil, mountains, forests)
    - Flora and fauna (plants and animals)
    """,
    tools=[textshow],
    handoff_description="Specialized in natural environment",
    model=OpenAIChatCompletionsModel(model="gemini-1.5-flash", openai_client=provider)
)

BuiltEnv_Agent = Agent(
    name="BuiltEnv Agent",
    instructions="""
    Answer all questions related to the built environment, including:
    - Buildings, cities, roads
    - Infrastructure (dams, bridges, transportation)
    - Technology and industry
    """,
    tools=[textshow],
    handoff_description="Specialized in built environment",
    model=OpenAIChatCompletionsModel(model="gemini-1.5-flash", openai_client=provider)
)

SocialEnv_Agent = Agent(
    name="SocialEnv Agent",
    instructions="""
    Answer all questions related to the social environment, including:
    - Communities and traditions
    - Laws, education, economy
    - Lifestyle and behavior patterns
    """,
    tools=[textshow],
    handoff_description="Specialized in social environment",
    model=OpenAIChatCompletionsModel(model="gemini-1.5-flash", openai_client=provider)
)
CompleteAgent = Agent(
    name="Complete Agent",
    instructions=(
        "You are a highly knowledgeable AI assistant capable of answering any question "
        "on any topic, including coding, websites, apps, chatbots, science, history, "
        "culture, current events, and more. Always provide detailed, accurate, and "
        "comprehensive answers. If information is not known for certain, provide the "
        "best possible informed response without refusing to answer or saying you lack data."
    ),
    tools=[textshow],  # Include any useful tools you have
    handoff_description="Expert at answering any question thoroughly on any topic.",
    model=OpenAIChatCompletionsModel(model="gemini-1.5-flash", openai_client=provider)
)



triage_Agent = Agent(
    name="Triage_Agent",
    instructions="Select which agent to use based on the user input.",
    handoffs=[NatEnv_Agent, BuiltEnv_Agent, SocialEnv_Agent, CompleteAgent],
    tools=[textshow],
    input_guardrails=[InputGuardrail(guardrail_function=EnviromentGuardrail)],
    model=OpenAIChatCompletionsModel(model="gemini-1.5-flash", openai_client=provider)
)

async def outview(user_input: str):
    result = await Runner.run(triage_Agent, input=user_input)  # pass raw string
    return result.final_output

if __name__ == "__main__":
    asyncio.run(outview(user_input))


