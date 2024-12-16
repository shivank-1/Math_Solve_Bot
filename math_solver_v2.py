import streamlit as st
import os
import sympy as sp
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler

# Load environment variables
load_dotenv()
groqkey = os.getenv('GROQKEY')

# Streamlit configuration
st.set_page_config(page_title="Text to Math Problem Solver and Data Search Assistant", page_icon="🤖")
st.title("Text To Math Problem Solver Using Gemma 2")

# Initialize Groq LLM
llm = ChatGroq(model="llama3-8b-8192", groq_api_key=groqkey)

# Custom function for symbolic math operations
def symbolic_math_solver(query):
    try:
        # Variable handling
        x = sp.Symbol('x')
        
        # Handle different types of mathematical operations
        if 'integrate' in query.lower():
            # Extract the function to integrate
            func_str = query.split('integrate(')[1].split(')')[0].strip()
            func = sp.sympify(func_str)
            result = sp.integrate(func, x)
            return f"Integration result: {result}"
        elif 'derivative' in query.lower() or 'diff' in query.lower():
            # Extract the function to differentiate
            func_str = query.split('diff(')[1].split(')')[0].strip()
            func = sp.sympify(func_str)
            result = sp.diff(func, x)
            return f"Derivative result: {result}"
        else:
            # Fallback to regular math chain
            math_chain = LLMMathChain.from_llm(llm=llm)
            return math_chain.run(query)
    except Exception as e:
        return f"Error solving mathematical problem: {str(e)}"

# Wikipedia Search Tool
Wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=Wikipedia_wrapper.run,
    description="A tool for searching the internet to find various information on the topic mentioned"
)

# Math Tool with custom symbolic math solver
calculator = Tool(
    name="Calculator",
    func=symbolic_math_solver,
    description="A tool for solving mathematical problems, including integration, differentiation, and numerical calculations"
)

# Reasoning Prompt Template
prompt = '''
You are an agent tasked with solving users' mathematical questions. Logically arrive at the solution and provide a detailed expression 
and display it point-wise for the question below.
Question: {question}
'''
prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

# Reasoning Chain
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions"
)

# Initialize Assistant Agent
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handling_parsing_errors=True
)

# Session State Management
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I am a Math Chatbot who can answer all your Math problems"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# User Interaction
question = st.text_area("Enter your question:", "integration of 4x^2")

if st.button("Let's Go!"):
    if question:
        with st.spinner("Generating response..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write('### Response:')
            st.success(response)
    else:
        st.warning("Please enter your question")