from langchain_core.documents import Document
from dotenv import load_dotenv
from commons import load_markdown_file

import autogen
import os

load_dotenv()
llm_config = {
    "cache_seed": 1,
    "temperature": 0,
    "config_list": [
        {
            "model": "gpt-4o",
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    ],
    "timeout": 360
}

data: Document = load_markdown_file("./templates/template-description.md")

## Init Agents
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    system_message = "A Human Head responsible for the task",
    code_execution_config={
        "last_n_messages": 2,
        "work_dir": "groupchat",
        "use_docker": False,
    },  
    human_input_mode="NEVER",
)

solutions_architect = autogen.AssistantAgent(
    name = "solutions_architect", 
    system_message = """
    **Role**: You are a solutions architect. You need to develop architecture proposals based on the needs of the clients.
    your proposals should give a layer 1 and 2 from the C4-Model considering the necessity and the FinOps.
    You should generate at least 3 approaches to solve the situation. At the end, state the advantages and disadvantages
    from each approach.
    
    **Example of one approach response**:
    Layer 1:
    System: Online Retail Application

    External Users:
        Customers: Browse products, place orders, and manage profiles.
        Administrators: Manage inventory, and process orders.
        
    External Systems:
        Payment Gateway: Processes customer payments.
        Email Service: Sends order confirmations and notifications.
        Shipping Service: Manages shipment tracking and delivery.
        
    Layer 2:
    
        Web Application: Serves the front end for customers and administrators.
        Mobile Application: Provides a mobile interface for customers.
        API Server: Handles business logic and processes requests from the web and mobile applications.
        Database: Stores product information, user profiles, orders, etc.
        Message Broker: Facilitates communication between services and external systems.
        Admin Console: Dedicated interface for administrators to manage inventory and orders.
    """,
    llm_config={"config_list": llm_config["config_list"]}
)

lead_architect = autogen.AssistentAgent(
    name="lead_architect",
    system_message="""
        **Role**: You are a lead Architect tasked with managing the architects. Each architect
        will perform a diferent task and respond with their results. You will critically review
        those and choose which approach considering advantages and disadvantages of each strategy.
        Choose the best solution in accordance with the business requirements and architecture best practices.
        
        **Team**: Your team have a solutions architect and a aws cloud architect, the solutions architect will create
        the solutions and the c1 and c2 from the c4-model, you need to choose the best approach given. The Cloud architect
        will give- you a diagram as a code using Diagrams, you will review it and ask for changes when needed.
        
        **IMPORTANT**: You can have 3 actions in your response: CLOUD_ARCHITECT_FOLLOW, SOLUTIONS_ARCHITECT_FOLLOW or TERMINATE. 
        If you think that solutions architect need to change something,
        write SOLUTIONS_ARCHITECT_FOLLOW at the end of your response.
        If you think that cloud architect need to change something,
        write SOLUTIONS_ARCHITECT_FOLLOW at the end of your response.
        If you think that the task is good enough, write TERMINATE at the end of your response.
        
    """
)

aws_cloud_architect = autogen.AssistentAgent(
    name="aws_cloud_architect",
    system_message="""
        **Role**: You are an expert aws cloud architect. You need to develop architecture proposals 
        using AWS cloud, you will receive at  and provide a data architecture for each. At the end, briefly state the advantages of cloud over on-premises 
        architectures, and summarize your solutions for each cloud provider using a table for clarity.
        You will give responses creating diagrams as a code.
    """,
    llm_config=llm_config
)

## Define graph-function
def conversation_graph(last_speaker, groupchat):
   messages = groupchat.messages

   if last_speaker is user_proxy:
       return solutions_architect
   elif last_speaker is solutions_architect:
       return lead_architect
   elif last_speaker is aws_cloud_architect:
       return lead_architect
   elif last_speaker is lead_architect:
        if "SOLUTIONS_ARCHITECT_FOLLOW" in messages[-1]["content"]:
           return solutions_architect
        elif "CLOUD_ARCHITECT_FOLLOW" in messages[-1]["content"]:
            return aws_cloud_architect
        elif "TERMINATE" in messages[-1]["content"]:
            return None


groupchat = autogen.GroupChat(
    agents=[user_proxy, aws_cloud_architect, lead_architect, solutions_architect], 
    messages=[], 
    max_round=15,
    speaker_selection_method=conversation_graph,
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

user_proxy.initiate_chat(
    manager, message="Provide your best architecture for a platform team, that needs to provide development teams a deployment resiliency, high velocity on change management"
)