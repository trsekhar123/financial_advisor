import os
from dotenv import load_dotenv

from groq import Groq
load_dotenv()
from langchain_community.tools.tavily_search import TavilySearchResults
import json
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import pandas as pd
from io import StringIO


memory = SqliteSaver.from_conn_string(":memory:")

groq_api_key = os.getenv("GROQ_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")
# client = Groq(api_key=groq_api_key)

# llm_name = "llama-3.3-70b-versatile"
llm_name = "llama3-groq-70b-8192-tool-use-preview"

from langchain_groq import ChatGroq

model = ChatGroq(
    model=llm_name,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

from tavily import TavilyClient

tavily_obj = TavilyClient(api_key=tavily_key)

from typing import TypedDict, List
from langchain_core.pydantic_v1 import BaseModel


class AgentState(TypedDict):
    task: str
    ticker: str
    # competitors: List[str]
    csv_data: str
    financial_data: str
    analysis: str
    # competitor_data: str
    recommendation: str
    feedback: str
    report: str
    content: List[str]
    revision_number: int
    max_revisions: int

class Queries(BaseModel):
    queries: List[str]

class StockTicker(BaseModel):
    ticker: str

GET_STOCK_TICKER = """You are an expert financial analyst. Get the stock ticker that is compatible with yfinance for the given task. JUST GIVE THE STOCK TICKER, NO EXTRA INFORMATION.
Sample Stocks and their tickers are given here.{csv_data}"""
GATHER_FINANCIAL_REPORT = """You are an expert financial analyst. Gather the financial data for the given company. Provide detailed financial data."""
ANALYSE_DATA_PROMPT = """You are an expert financial analyst. Analyse the provided financial data and provide detailed analysis and insights. """
RESEARCH_COMPETITORS_REPORT = """You are a researcher tasked with providing information about company sentiments, FIIs, DIIs interest and other hot topics. Generate a list of search queries to gather relevant information. Only generate 3 queries max."""
COMPETITOR_PERFORMANCE_REPORT = """You are an expert financial analyst. Analyse financial performance of the given company with its sentiments and fundamentals based on the provided data
***MAKE SURE TO INCLUDE THE RECOMMENDATIONS IN THE RECOMMENDATION***"""
FEEDBACK_PROMPT = """You are reviewer. Provide detailed feedback and critique for the provided financial report. Include any additional information or revisions provided"""
WRITE_REPORT_PROMPT = """You are a financial stock recommender. Give a comprehensive financial stock recommendation based on analysis, research, fundamental analysis, and feedback provided. The recommendation should be Buy, Sell or Hold. If there is blood in the street and fundamentals are very good, then Buy"""
RESEARCH_CRITIQUE_PROMPT = """You are a researcher tasked with providing information to address the provided critique. Generate a list of search queries to gather relevant information. Only generate 3 queries max."""

def get_stock_ticker(state: AgentState):
    task = state['task']
    csv_data = state['csv_data']

    messages = [
        SystemMessage(content=GET_STOCK_TICKER.format(csv_data=csv_data)),
        HumanMessage(content=task)
    ]

    response = model.with_structured_output(StockTicker).invoke(messages)

    ticker =  response.ticker
    ticker =  ticker if ticker.endswith(".NS") else ticker+".NS"

    print(response)

    return {"ticker":ticker}


def gather_financial_data(state:AgentState):
    import yfinance as yf

    # Define the stock ticker
    ticker_symbol = state['ticker'].strip()  # Example: NSE-listed Reliance Industries
    print(ticker_symbol)
    # Fetch the stock data
    stock = yf.Ticker(ticker_symbol)

    # print(stock.info)

    # Current Price
    current_price = stock.history(period="1d")["Close"].iloc[-1]

    # Fetch fundamentals
    fundamentals = stock.info



    # Get last 1-year weekly price data
    weekly_prices = stock.history(period="1y", interval="1wk")

    fundamentals_data = {
        "Current Price": stock.history(period="1d")["Close"].iloc[-1],
        "Market Cap": fundamentals.get("marketCap", "N/A"),
        "Trailing P/E": fundamentals.get("trailingPE", "N/A"),
        "Forward P/E": fundamentals.get("forwardPE", "N/A"),
        "PEG Ratio": fundamentals.get("pegRatio", "N/A"),
        "P/B Ratio": fundamentals.get("priceToBook", "N/A"),
        "EPS (Trailing)": fundamentals.get("trailingEps", "N/A"),
        "Revenue Growth (YoY)": fundamentals.get("revenueGrowth", "N/A"),
        "Earnings Growth": fundamentals.get("earningsGrowth", "N/A"),
        "Profit Margin": fundamentals.get("profitMargins", "N/A"),
        "Debt-to-Equity": fundamentals.get("debtToEquity", "N/A"),
        "Current Ratio": fundamentals.get("currentRatio", "N/A"),
        "Quick Ratio": fundamentals.get("quickRatio", "N/A"),
        "Dividend Yield": fundamentals.get("dividendYield", "N/A"),
        "Payout Ratio": fundamentals.get("payoutRatio", "N/A"),
        "Book Value Per Share": fundamentals.get("bookValue", "N/A"),
        "Enterprise Value": fundamentals.get("enterpriseValue", "N/A"),
        "EV/EBITDA": fundamentals.get("enterpriseToEbitda", "N/A"),
    }

    # Print extracted data
    finance_data = ""
    for key, value in fundamentals_data.items():
        finance_data+=f"{key}: {value}\n"

    # print("Last 1-Year Weekly Prices:")
    # print(weekly_prices.tail())



    convert_financial_data = weekly_prices.tail(10).to_string(index=False)

    combined_content = (
        f"{state["task"]}\n\n Here is the weekly price data for last 10 weeks:\n\n{convert_financial_data}\n\n Here is the stock fundamentals{finance_data}"
    )

    messages = [
        SystemMessage(content=GATHER_FINANCIAL_REPORT),
        HumanMessage(content=combined_content)
    ]

    response = model.invoke(messages)

    return {"financial_data":response.content}

def analyze_datanode(state: AgentState):
    messages = [
        SystemMessage(content=ANALYSE_DATA_PROMPT),
        HumanMessage(content=state['financial_data'])
    ]
    response = model.invoke(messages)
    return {"analysis":response.content}

def research_competitors_node(state: AgentState):
    content = state.get("content",[])

    
    queries = model.with_structured_output(Queries).invoke(
        [
            SystemMessage(content=RESEARCH_COMPETITORS_REPORT),
            HumanMessage(content=state['analysis'])
        ]
    )

    for q in queries.queries:
        response = tavily_obj.search(query=q, max_results = 2)
        for r in response["results"]:
            content.append(r["content"][:150])

    return {"content":content}


def compare_performace_node(state:AgentState):
    content = "\n\n".join(state.get("content",[]))

    user_message = HumanMessage(
        content = f"{state['task']}\n\n Here is the financial analysis:\n\n {state['analysis']}\n\n{content}"
    )

    messages = [
        SystemMessage(content=COMPETITOR_PERFORMANCE_REPORT),
        user_message
    ]

    response=model.invoke(messages)

    return {
        "recommendation":response.content,
        "revision_number":state.get("revision_number",1) + 1
    }

def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke(
        [
            SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=state["feedback"])
        ]
    )

    content = state.get("content",[])

    for q in queries.queries:
        response = tavily_obj.search(query=q, max_results=2)
        for r in response["results"]:
            content.append(r["content"][:100])

    return {'content':content}

def collect_feedback_node(state:AgentState):
    messages = [
        SystemMessage(content=FEEDBACK_PROMPT),
        HumanMessage(content=state["recommendation"])
    ]

    response=model.invoke(messages)

    return {'feedback':response.content}

def write_report(state: AgentState):
    messages = [
        SystemMessage(content=WRITE_REPORT_PROMPT),
        HumanMessage(content=state['recommendation'])
    ]
    response=model.invoke(messages)
    return {'report':response.content}

def should_continue(state):
    if state['revision_number'] > state['max_revisions']:
        return END
    return "collect_feedback"


builder = StateGraph(AgentState)

builder.add_node("get_stock_ticker",get_stock_ticker)
builder.add_node("gather_financials",gather_financial_data)
builder.add_node("analyze_data",analyze_datanode)
builder.add_node("research_competitors",research_competitors_node)
builder.add_node("compare_performance",compare_performace_node)
builder.add_node("collect_feedback",collect_feedback_node)
builder.add_node("research_critique",research_critique_node)
builder.add_node("write_report",write_report)


builder.set_entry_point("get_stock_ticker")

builder.add_conditional_edges("compare_performance",
                              should_continue,
                              {END:END, "collect_feedback":'collect_feedback'})

builder.add_edge("get_stock_ticker","gather_financials")
builder.add_edge("gather_financials","analyze_data")
builder.add_edge("analyze_data","research_competitors")
builder.add_edge("research_competitors","compare_performance")
builder.add_edge("collect_feedback","research_critique")
builder.add_edge("research_critique","compare_performance")
builder.add_edge("compare_performance","write_report")

# graph = builder.compile(checkpointer=memory)

# # ============ For Console Testing ===================
# def read_csv_file(file_path):
#     with open(file_path,"r") as fp:
#         print("Reading csv file")
#         return fp.read()
    

# if __name__=='__main__':
#     task = "Is it good to buy IndusInd Bank?"
#     # competitors = ["Microsoft","Nvidia","Google"]
#     csv_file_path = (
#         'financials_data.csv'
#     )

#     if not os.path.exists(csv_file_path):
#         print(f"CSV file path not found at {csv_file_path}")
#     else:
#         print("Starting conversation")
#         import pandas as pd
#         csv_data = pd.read_csv(csv_file_path).to_string(index=False)
        
#     initial_state = {
#         "task":task,
#         # "competitors":competitors,
#         "csv_data":csv_data,
#         "max_revisions":2,
#         "revision_number":1
#     } 
#     thread_config = {
#         "configurable" : {"thread_id":"1"}
#     }

#     with SqliteSaver.from_conn_string(":memory:") as checkpointer:
#         graph = builder.compile(checkpointer=checkpointer)
#         for s in graph.stream(initial_state,thread_config):
#             print(s)

# #============== streamlit ui =============
import streamlit as st

def main():
    st.title("Finanacial performance reporting Agent")

    task = st.text_input(
        "Enter the task:",
        "Analyse the financial performance of our company (MyAICo) compared to competitors."
    )
    # competitors = st.text_area("Enter the competitors names(one per line):").split("\n")
    max_revisions = st.number_input("Max revisions", min_value=1)
    # upload_file = st.file_uploader(
    #     "Upload a csv file with the company's financial data", type=["csv"]
    # )
    csv_file_path = (
        'stock_tickers.csv'
    )

    if st.button("Start Analysis") and task is not None:
        csv_data = pd.read_csv(csv_file_path).to_string(index=False)#.getvalue().decode("utf-8")
        # st.write(csv_data)

        initial_state = {
            "task":task,
            # "competitors":[comp.strip() for comp in competitors],
            "csv_data":csv_data,
            "max_revisions":max_revisions,
            "revision_number":1
        } 
        thread_config = {
            "configurable" : {"thread_id":"1"}
        }
        final_state = None
        with SqliteSaver.from_conn_string(":memory:") as checkpointer:
            graph = builder.compile(checkpointer=checkpointer)
            for s in graph.stream(initial_state,thread_config):
                # st.write(s)
                final_state = s
            
            if final_state and "report" in final_state:
                st.subheader("Final State")
                st.markdown(final_state["report"])


if __name__ == "__main__":
    main()


# #============== streamlit ui end =========