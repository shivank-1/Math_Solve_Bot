import streamlit as st
import os
import sympy as sp
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent, AgentExecutor
from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler

# Load environment variables
load_dotenv()
groqkey = os.getenv('GROQKEY')

# Streamlit configuration
st.set_page_config(page_title="Text to Math Problem Solver and Data Search Assistant", page_icon="🤖")
st.title("Text To Math Problem Solver Using Gemma 2")

# Initialize Groq LLM with enhanced parameters
llm = ChatGroq(
    model="llama3-8b-8192", 
    groq_api_key=groqkey,
    temperature=0.3,  # Lower temperature for more focused responses
    max_tokens=2000   # Increase maximum token limit
)

# Improved custom function for symbolic math operations
def symbolic_math_solver(query):
    try:
        # More robust handling of mathematical queries
        x = sp.Symbol('x')
        
        # Improved parsing for integration and differentiation
        import re
        if 'integrate' in query.lower():
            match = re.search(r'integrate\((.*?)\)', query)
            if match:
                func_str = match.group(1).strip()
                func = sp.sympify(func_str)
                result = sp.integrate(func, x)
                return f"Integration result: ∫({func_str})dx = {result}"
        
        elif 'derivative' in query.lower() or 'diff' in query.lower():
            match = re.search(r'diff\((.*?)\)', query)
            if match:
                func_str = match.group(1).strip()
                func = sp.sympify(func_str)
                result = sp.diff(func, x)
                return f"Derivative result: d/dx({func_str}) = {result}"
        
        # Enhanced fallback mechanism
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
    description="A tool for solving advanced mathematical problems, including integration, differentiation, and numerical calculations"
)

# Enhanced Reasoning Prompt Template
prompt = '''
You are an advanced mathematical reasoning agent. For the given mathematical question:
1. Carefully analyze the problem
2. Break down the solution into clear, logical steps
3. Provide a comprehensive and precise solution
4. Explain the reasoning behind each step

Question: {question}

Provide a detailed, step-by-step solution with clear mathematical reasoning.
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
    description="A sophisticated tool for solving complex mathematical reasoning problems with detailed explanations"
)

# Create the agent with proper configuration
agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,  # Enable verbose mode for debugging
    handle_parsing_errors=True
)

# Wrap the agent in an AgentExecutor with specific parameters
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent.agent,
    tools=[wikipedia_tool, calculator, reasoning_tool],
    max_iterations=5,
    early_stopping_method="generate",
    handle_parsing_errors=True
)

# Session State Management
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I am an Advanced Math Reasoning Assistant ready to solve your mathematical problems!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# User Interaction
question = st.text_area("Enter your mathematical question:", "integration of 4x^2")

if st.button("Solve Problem"):
    if question:
        with st.spinner("Analyzing and solving the mathematical problem..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            try:
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
                # Use agent_executor.run instead of agent.run
                response = agent_executor.run(input=question, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write('### Solution:')
                st.success(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a mathematical question")