
import json
import os
from datetime import datetime

import yfinance as fy

from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults, BraveSearch


from IPython.display import Markdown

# %%
def fetch_stock_price(ticket):
    stock = fy.download(ticket, start = "2023-08-08", end="2024-08-08")
    return stock

yahoo_finance_tool = Tool(
    name = "Yahoo Finance Tool",
    description = "Fetches stock prices for {ticket} from the last year about a specific stock from Yahoo Finance API",
    func = lambda ticket: fetch_stock_price(ticket)
)

# %%
os.environ['OPENAI_API_KEY'] == st.secrets['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-3.5-turbo")


# %%
stockPriceAnalisty = Agent(
    roles = "Senior stock price analisty",
    goal = "find the {ticket} stok price and analyses trends",
    backstory = """ you`re hightly experienced in analysing the price of an a specific stock
    and make predictions about it`s future price""",
    verbose = True,
    llm = llm,
    max_iter = 5,
    memory = True,
    tools = [yahoo_finance_tool]
)

# %%
getStockPrice = Task(
    description = "Analyse the stock {ticket} price history and create a trend analyses of up, down or sideways",
    expected_output = """ Specify the current trend stock price - up, down or sideways.
    eg. stock='AAPL', price UP """,
    agent = stockPriceAnalisty
)

# %%
search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)

# %%
newAnalist = Agent(
    roles = "Stock News Analisty",
    goal = """Create a short sumary of the market news related to {ticket} company. Specify the current trend - up, down or sideways with
    the news context. For each request stock assert, specify a numbet between 0 and 100, where 0 is extreme fear and 100 is extreme greed.""",
    backstory = """ you`re hightly experienced in analysing the market trends and news and have trackeds assert for more then 10 years.
    you`re also master level analyts in the tradicional markets and deep understanding of human psycology.

    You understand news, theirs tittles and information, but you look at those with a health dose skeptism. You consider also the source of the news articles
     
    """,
    verbose = True,
    llm = llm,
    max_iter = 10,
    memory = True,
    tools = [yahoo_finance_tool],
    allow_delegation = False 
)

# %%
get_new = Task(
    description = """Take the stock and always include BTC to it (if not request)
    Use the search tool to search each one individualy.
    
    the current date is {datetime.now()}.
    
    compose the result into a helpfull report 
    """,
    expected_output = """A summary of the overeall market and one sentence summary for each request asset.
    Include a fear/greed score for each asset based on the news. Use format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE>    """,
    agent = newAnalist
)

# %%
stockAnalistyWrite = Agent(
    role = "Senior Stock Analyst Writer",
    goal = """ Analyse trend prices and news and write an insighfull compelling and informative 3 paragraph long newletters based on the stock report and price trend.""",
    backstory = """ You`re widely accepted as the best stock analyst in the market. You understand complex concepts and create compelling stories and narratives
    that resonate with wider audiences 
    you understand macro factors and combine multiples theories - eg. cyvle theory and fundamental analyses. You`re able to hold multipli opinions when analyzing anything.
    """,
    verbose = True,
    llm = llm,
    max_iter = 5,
    memory = True,
    allow_delegation = True
)

# %%
writeAnalyses = Task(
    description = """Use the stock prices trend and the stock news report to create an analyses and write the newletter about the {ticket} company
    that is brief and highlights the most important points.
    Focus on the stock price trend, news and fear/greed score. What are the near future considerations?
    Include the previous analyses of stock trend and news sumary.
    """,
    expected_output="""
    An eloquent 3 paragraphs newletter formated as markdow in a easy readable manner. It should contains: 
    -3 bullets executives summary
    -Introduction - set the overrall pictures and spike up the interest 
    -main part provides the meat of the analyst incluiding the news sumary and fead/greed scores
    -sumary - keys facts and concrete future trend prediction - up, down or sideways.
""",
    agent = stockAnalistyWrite,
    context = [getStockPrice, get_new]
)

# %%
crew = Crew(
    agents = [stockPriceAnalisty, newAnalist, stockAnalistyWrite],
    tasks = [getStockPrice, get_new, writeAnalyses],
    verbose = 2,
    process = Process.hierarchical,
    full_output = True,
    share_crew = False,
    manager_llm = llm,
    max_iter = 15
)


with st.siderbar:
    st.header('Enter the ticket stack')

    with st.form(key='research_form'):
        topic = st.text_input("Select the ticket")
        submit_buttom = st.form_submit_buttom(label = "Run Research")


if submit_buttom:
    if not topic:
        st.error("Please fill the ticket field")
    else:
        result= crew.kickoff(inputs={'ticket': topic})

        st.subheader("Results of research:")
        st.write(results['final_output'])
