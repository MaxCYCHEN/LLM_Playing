import gradio as gr
import os
import time
from pathlib import Path

from langchain.vectorstores import FAISS
#from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
import json

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

#from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI

#from langchain import PromptTemplate 
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
#from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings

from langchain.callbacks import get_openai_callback

from langchain.tools.retriever import create_retriever_tool
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_tools_agent

import APIKEY

#import lancedb
#from langchain.vectorstores import LanceDB

# SSL Issue
#os.environ['REQUESTS_CA_BUNDLE'] = "cacert.pem"
class EvaluateMathExpression(BaseTool):
    name = "Calculator"
    description = "Use this tool to evaluate a math expression."
    def _run(self, expr: str):
        return eval(expr)
    def _arun(self, query: str):
        raise NotImplementedError("Async not supported")

class QA_LangChain():
    def __init__(self, app_type, chip_type):
        self.llm = None
        self.tools = None
        self.memory = None
        #self.gptmodel = "gpt-3.5-turbo-1106" #"gpt-3.5-turbo"
        #self.gptmodel = "gpt-4"
        self.gptmodel = "gpt-4-1106-preview" #"gpt-4-0125-preview"
        self.mmr_num = 8

        self.app_type = app_type.upper()
        self.chip_type = chip_type.upper()

        # 03/12 Terry request
        if (self.chip_type in ['M463', 'M467']):
            self.prompt_chip_type = 'M463_M467'
        elif (self.chip_type in ['M251', 'M252', 'M254', 'M256', 'M258']):
            self.prompt_chip_type = 'M251_M252_M254_M256_M258' #'M251_M252_M254_M256_M258'
        elif (self.chip_type in ['M253']):
            self.prompt_chip_type = 'M253'
        elif (self.chip_type in ['M031', 'M032']):
            self.prompt_chip_type = 'M031_M032'    
        elif (self.chip_type in ['NUC100', 'NUC200']):
            self.prompt_chip_type = 'NUC100_NUC200'

        if self.app_type == 'QA':
        # Create the chat prompt templates
            system_template = """You are a technical engineer to answer chip or math relative question at the end.
You can access three tools, the first is searching Nuvoton """+self.prompt_chip_type+""" Series Technical Reference Manual.
If you can't get enough infomation, use the second tool, searching tables in Technical Reference Manual.
The third tool is a calculator. Organize the context from these tools to finish the answer.
If you don't know the answer or the question has nothing to do with technical, don't try to make up an answer.
"""    
        if self.app_type == 'QA_SVD_1.1':
        # Create the chat prompt templates
            system_template = """You are a technical engineer to answer chip or math relative question at the end.
You can access three tools, the first is searching Nuvoton """+self.prompt_chip_type+""" Series Technical Reference Manual.
If you can't get enough infomation, use the second tool, searching XML format system view description tool. 
The third tool is a calculator. Organize the context from these tools to finish the answer.
If you don't know the answer or the question has nothing to do with technical, don't try to make up an answer.
"""
        elif self.app_type == 'CODEGEN':
        # Create the chat prompt templates
            system_template = """You are a C software and firmware engineer, and good at produce a C code basing on anyone's question.
Use the following two tools, the first is searching the standard driver C header and source files of """+self.chip_type+""" MCU.
Try your best to use the APIs, functions, defines and registers in this standard driver C header and source files tool.
If you need math calculation, the second tool is a calculator. Organize the context from these tools to finish the C code.
If you don't know the answer or the question has nothing to do with technical, don't try to make up an answer.
"""
        
        messages =[("system", system_template),
            MessagesPlaceholder(variable_name='chat_history', optional=True),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name='agent_scratchpad')]
        
        self.qa_prompt = ChatPromptTemplate.from_messages(messages)

    def create_ag_retriver(self, embeddings, input_doc_pth):
        if self.app_type == "QA":
            # FAISS  
            vectorstore=FAISS.load_local(input_doc_pth, embeddings)
            retriever_vec=vectorstore.as_retriever(
            search_type="mmr", # Also test "similarity"
            search_kwargs={"k": self.mmr_num})

            # FAISS Add table retriver
            input_doc_table_pth = (str)(Path(input_doc_pth).parent / (str)(Path(input_doc_pth).name).replace("pypdf", "table", 1))
            # logical parent
            vectorstore_table=FAISS.load_local(input_doc_table_pth, embeddings)
            retriever_vec_table=vectorstore_table.as_retriever(
            search_type="mmr", # Also test "similarity"
            search_kwargs={"k": 10})

            return  [retriever_vec, retriever_vec_table]
        
        if self.app_type == "QA_SVD_1.1":
            # FAISS  
            vectorstore=FAISS.load_local(input_doc_pth, embeddings)
            retriever_vec=vectorstore.as_retriever(
            search_type="mmr", # Also test "similarity"
            search_kwargs={"k": self.mmr_num})

            # FAISS Add SVD retriver
            input_svd_pth = (str)(Path(input_doc_pth).parent / Path(r"SVD_" + self.prompt_chip_type))
            # logical parent
            vectorstore_table=FAISS.load_local(input_svd_pth, embeddings)
            retriever_vec_svd=vectorstore_table.as_retriever(
            search_type="mmr", # Also test "similarity"
            search_kwargs={"k": 8})

            return  [retriever_vec, retriever_vec_svd]
        
        elif self.app_type == "CODEGEN":
            # FAISS  
            vectorstore=FAISS.load_local(input_doc_pth, embeddings)
            retriever_vec=vectorstore.as_retriever(
            search_type="mmr", # Also test "similarity"
            search_kwargs={"k": 10})

            return  [retriever_vec]
            

    def create_ag_tools(self, retriever_list):
        if self.app_type == "QA":
            tool_trm_search = create_retriever_tool(
                retriever=retriever_list[0],
                name = "search_{}_technical_reference_manual".format(self.chip_type),
                description = "Searches and returns detail from the {} technical reference manual.".format(self.prompt_chip_type),
            )
            tool_trm_table_search = create_retriever_tool(
                retriever=retriever_list[1],
                name = "search_{}_technical_reference_manual_tables".format(self.chip_type),
                description = "Searches and returns detail from the {} tables in technical reference manual.".format(self.prompt_chip_type),
            )
            self.tools = [tool_trm_search, tool_trm_table_search, EvaluateMathExpression()]

        if self.app_type == "QA_SVD_1.1":
            tool_trm_search = create_retriever_tool(
                retriever=retriever_list[0],
                name = "search_{}_technical_reference_manual".format(self.chip_type),
                description = "Searches and returns detail from the {} technical reference manual.".format(self.prompt_chip_type),
            )
            tool_svd_search = create_retriever_tool(
                retriever=retriever_list[1],
                name = "search_{}_XML_format_system_view_description".format(self.chip_type),
                description = "Searches and returns detail from the {} XML format system view description which contains registers information.".format(self.prompt_chip_type),
            )
            self.tools = [tool_trm_search, tool_svd_search, EvaluateMathExpression()]

        elif self.app_type == "CODEGEN":
            tool_CCODE_search = create_retriever_tool(
                retriever=retriever_list[0],
                name = "search_{}_standard_driver_C_code".format(self.chip_type),
                description = "Searches and returns detail from the {} standard driver C header and source files.".format(self.chip_type),
            )
            self.tools = [tool_CCODE_search, EvaluateMathExpression()]
            
    def LangChain_RQA_agent(self, input_doc_pth):
        # load embedding model
        print("===== Load the embedding model =====")
        
        # choose your embeddings model
        #embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device': 'cpu'})
        #embeddings = HuggingFaceInstructEmbeddings(model_name='BAAI/bge-small-en',model_kwargs={'device': 'cpu'})
        embeddings = OpenAIEmbeddings()

        print("===== Load the FAISS =====")
        retriever_list = self.create_ag_retriver(embeddings, input_doc_pth)
     
        print("===== Create the Tools =====")
        self.create_ag_tools(retriever_list)

        print("===== Create the Agent =====")
        return self.create_agent()
    
    def create_agent(self):
        agent = create_openai_tools_agent(self.llm, self.tools, self.qa_prompt)
        return AgentExecutor(agent=agent, tools=self.tools)
    
    def QA_LangChain_RQA_model_Chatgpt_create(self):
        os.environ["OPENAI_API_KEY"] = APIKEY.API_KEY_SERVICE_OPENAI
        self.llm = ChatOpenAI(temperature=0, model=self.gptmodel)

    # Adding LangSmith
    def langsmith_create(self, smith_eable, app_type, chip_type):
        if smith_eable == "Enable":
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = "RAGagent_Gradio_{}_{}".format(app_type, chip_type)
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
            os.environ["LANGCHAIN_API_KEY"] = APIKEY.LANGCHAIN_API_KEY  # Update to your API key
        else:
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            os.environ["LANGCHAIN_PROJECT"] = "RAGagent_Gradio_{}_{}".format(app_type, chip_type)
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
            os.environ["LANGCHAIN_API_KEY"] = APIKEY.LANGCHAIN_API_KEY  # Update to your API key    

        from langsmith import Client
        client = Client()


class QA_UI():
    def __init__(self):
        self.qa_doc_dir = r'doc_faiss'
        
        self.agent = None
        self.doc_path = os.path.join(self.qa_doc_dir, r'M463_M467_bsp_StdDriver_Regs')
        
        # We need to initial the llm model when intial the class & can't create another
        self.QA_LangChain_CLS = None
        #self.init_chat_qa_bot()

        # save the RQA's docs
        self.ref_docs = None
        self.return_msg = ""

        # save cb info for tokens
        self.cb_info = None
        self.tokens_str = ""

        self.chat_history_list = []

    #def init_chat_qa_bot(self):
    #    self.QA_LangChain_CLS = QA_LangChain()
    #   
    #    self.QA_LangChain_CLS.QA_LangChain_RQA_model_Chatgpt_create()
    #    self.QA_LangChain_CLS.langsmith_create(Path(self.doc_path).stem)
    #    self.agent = self.QA_LangChain_CLS.LangChain_RQA_agent(self.doc_path, self.QA_LangChain_CLS.qa_prompt)


    def QA_gradio_UI(self):

        def __chip_trm_select(app_type, chip_type):
            doc_path = None
            if app_type.upper() == "QA":
                root_path = os.path.join(self.qa_doc_dir, app_type.upper())
                files_dir = [f for f in Path(root_path).iterdir() if Path.is_dir(f)]
                for dir_pth in files_dir:
                    if (str)(Path(dir_pth).stem).find(chip_type.upper())!=-1 and (str)(Path(dir_pth).stem).find("pypdf")!=-1:  
                        doc_path = dir_pth
                        #print(doc_path)
                assert doc_path!=None, "{}: The chip_type: {} not support or missing the faiss doc! path:{}".format(app_type, chip_type, doc_path)
            if app_type.upper() == "QA_SVD_1.1":
                fix_app_type = "QA" # Use the doc_faiss/QA/ path
                root_path = os.path.join(self.qa_doc_dir, fix_app_type.upper())
                files_dir = [f for f in Path(root_path).iterdir() if Path.is_dir(f)]
                for dir_pth in files_dir:
                    if (str)(Path(dir_pth).stem).find(chip_type.upper())!=-1 and (str)(Path(dir_pth).stem).find("pypdf")!=-1:  
                        doc_path = dir_pth
                        #print(doc_path)
                assert doc_path!=None, "{}: The chip_type: {} not support or missing the faiss doc! path:{}".format(app_type, chip_type, doc_path)
    
            elif app_type.upper() == "CODEGEN":
                root_path = os.path.join(self.qa_doc_dir, app_type.upper())
                files_dir = [f for f in Path(root_path).iterdir() if Path.is_dir(f)]
                for dir_pth in files_dir:
                    if (str)(Path(dir_pth).stem).find(chip_type.upper())!=-1:  
                        doc_path = dir_pth
                        #print(doc_path)
                assert doc_path!=None, "{}: The chip_type: {} not support or missing the faiss doc! path:{}".format(app_type, chip_type, doc_path)

            return doc_path
        
        def start_qa_chain(app_type, chip_type, smith_eable):
            # So far because gradio interface, we can't clear the chatbox.
            # However, keeping the chat history is no harm. 
            #self.chat_history_list = []
            #print("create new chat history:")
            #print(self.chat_history_list)
            
            self.doc_path = __chip_trm_select(app_type, chip_type)
            
            self.QA_LangChain_CLS = QA_LangChain(app_type, chip_type)
            self.QA_LangChain_CLS.QA_LangChain_RQA_model_Chatgpt_create()
            self.QA_LangChain_CLS.langsmith_create(smith_eable, app_type, chip_type)
            self.agent = self.QA_LangChain_CLS.LangChain_RQA_agent(self.doc_path)

            print("create => agent detail:")
            print(self.agent)
  
            return "finish !"
        
        inputs = [gr.Dropdown(["QA", "CODEGEN", "QA_SVD_1.1"], label="Choose APP", info="", value="QA"),
                  gr.Dropdown(["M463", "M467", "M251", "M252", "M254", "M256", "M258", "M253", "M031", "M032"], label="Choose Chip Type", info="", value="M463"),
                  gr.Dropdown(["Enable", "Disable"], label="Upload to LangSmith", info="", value="Enable")
                 ]
        
        def return_ref_docs(): 
            for i in range(len(self.ref_docs)):
                if 'page' in self.ref_docs[i].metadata:
                    self.return_msg += "\n" + self.ref_docs[i].metadata['source'] + ":" + str((self.ref_docs[i].metadata['page']+1)) + ","
                else: 
                    self.return_msg += "\n" + self.ref_docs[i].metadata['source'] + ","
            self.return_msg += "\n" + "---------------------"         
            return self.return_msg
        
        def return_tokens():
            self.tokens_str += f"Total Tokens: {self.cb_info.total_tokens}"
            self.tokens_str += "\n" + f"Prompt Tokens: {self.cb_info.prompt_tokens}"
            self.tokens_str += "\n" + f"Completion Tokens: {self.cb_info.completion_tokens}"
            self.tokens_str += "\n" + f"Total Cost (USD): ${self.cb_info.total_cost}" 
            self.tokens_str += "\n" + f"---------------------" + "\n"          
            return self.tokens_str
        
        def return_ref_docs_clear():
                self.return_msg = ""
                return self.return_msg
        
        def return_tokens_clear():
                self.tokens_str = ""
                return self.tokens_str
         
        with gr.Blocks() as demo:    
            # Layout Section
            with gr.Tab("Agent ChatBot"):
                #gr.Markdown(value= "LLM model: {}".format(self.QA_LangChain_CLS.gptmodel))
                with gr.Row():
                    
                    gr.Interface(fn=start_qa_chain, inputs=inputs, outputs=None)
                    
                chatbot = gr.Chatbot(height = 600, show_copy_button = True, scale = 2)
                msg = gr.Textbox()
                ref_doc = gr.Textbox(label="Reference Documents",
                                     info="the number is page.",
                                     lines=1,
                                     value=" ")
                cb_tok_box = gr.Textbox(label="Tokens Summarize",
                                     lines=1,
                                     value=" ")

                with gr.Row():
                    enter_button = gr.Button("Enter")
                    save_button = gr.Button("Save")
                    clear = gr.Button("Clear")
                    #delete_button = gr.Button("delete model")
            
            #with gr.Tab("System Message Prompt"):
            #    gr.Markdown(
            #            """
            #            # Nuvoton-MCU QA and CodeGen Prompt
            #            #### Please update the LLM prompt template basing on your different task. For example:
            #            #### QA: 
            #            
            #            ```
            #            Use the following pieces of context and chat history to answer the question at the end. The context is Nuvoton M467 Series Technical Reference Manual.
            #            If you don't know the answer or the question has nothing to do with code or programing, don't try to make up an answer.
            #            ----------------
            #            {context}
            #            {chat_history}
            #            ```
#
            #            #### CodeGen: 
            #            ```
            #            Use the following pieces of context and chat history to answer the question at the end. The context is standard driver C header files of M460 MCU.
            #            Please answer with C Code function as complete as possible.
            #            If you don't know the answer or the question has nothing to do with code or programing, don't try to make up an answer.
            #            ----------------
            #            {context}
            #            {chat_history}
            #            ```
            #            """
            #    )
            #    prompt_inp = gr.Textbox(label="Prompt Edit", lines=4)
            #    #prompt_oup = gr.Textbox(label="Prompt Now", lines=4, placeholder=self.QA_LangChain_CLS.system_template)
            #    up_prompt_btn = gr.Button("Update Prompt")

                #@up_prompt_btn.click(inputs=prompt_inp, outputs=prompt_oup)
                #def update_prompt(prompt_inp):
                #    
                #    update_messages = [
                #    SystemMessagePromptTemplate.from_template(prompt_inp),
                #    HumanMessagePromptTemplate.from_template("{question}")
                #    ]
                #    self.QA_LangChain_CLS.qa_prompt = ChatPromptTemplate.from_messages(update_messages)
                #    
                #    # re-create llm chain with new prompt
                #    self.agent = self.QA_LangChain_CLS.LangChain_RQA_agent(self.doc_path,  self.QA_LangChain_CLS.qa_prompt)
                #    print("Re-create => ConversationalRetrievalChain detail:")
                #    print(self.agent)
#
                #    return prompt_inp
                ## update the prompt, so need to 1. rebuild the RQAchain, 2. clear the chatbox & info
                #up_prompt_btn.click(fn=update_prompt, inputs=prompt_inp, outputs=prompt_oup).then(
                #    lambda: None, None, chatbot, queue=False).then(
                #        return_ref_docs_clear, None, ref_doc).then(return_tokens_clear, None, cb_tok_box)
            
            #with gr.Tab("Documents (ToDo...)"):
            #    file_loader = gr.File(file_count="multiple", file_types=[".txt", ".pdf", ".csv"], height=100)
            #    state_text = gr.Textbox(label='model state:', value="Please load a document.")
            
            
            # Events Section
            def load_state_log():
                return "Load the document and model successfully"
            
            def upload_file(files):
                file_paths = [file.name for file in files]
                return file_paths
            
            def update_doc_and_reload_QA_sys(files):
                file_paths = [file.name for file in files]
                
            def user(user_message, history):
                return "", history + [[user_message, None]]
        
            def bot(history):
                # Get the user's Q msg
                # print(history[-1][0]) 
                # query = "What is the architecture of NuMicro M23"
                
                query = history[-1][0]
                with get_openai_callback() as cb:                                        
                    result= self.agent.invoke({"input": query, "chat_history": self.chat_history_list})
                    #print(result)

                    self.chat_history_list.append(HumanMessage(content = result['input']))
                    self.chat_history_list.append(AIMessage(content = result['output']))
                                                          
                    history[-1][1] = ""
                    history[-1][1] = result['output']
                    #print(history[-1][1])
    
                    #self.ref_docs = result['source_documents']
                    ##for i in range(len(self.ref_docs)):
                    ##    print(self.ref_docs[i].metadata['source'])

                    self.cb_info = cb

                return history

                #for character in bot_message:
                #    history[-1][1] += character
                #    time.sleep(0.05)
                #    yield history
                   
            def save_chat(history):
                with open('{}.json'.format(time.strftime("%Y%m%d-%H%M%S")), 'w', encoding='utf-8') as f:
                    json.dump(history, f, ensure_ascii=False, indent=4)

            def create_new_chain():
                # update to new chain, so the memory will be clean
                #self.memory.clear()
                #self.agent =  self.QA_LangChain_CLS.Create_Chain()

                self.chat_history_list = []
                print("create new chat history:")
                print(self.chat_history_list)
                
            # Interactive Section
            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, chatbot, chatbot).then(return_tokens, None, cb_tok_box)#.then(return_ref_docs, None, ref_doc)
            
            enter_button.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, chatbot, chatbot).then(return_tokens, None, cb_tok_box)#.then(return_ref_docs, None, ref_doc)
            
            clear.click(lambda: None, None, chatbot, queue=False).then(
                create_new_chain, None, None).then(return_tokens_clear, None, cb_tok_box)
            #.then(return_ref_docs_clear, None, ref_doc)
            
            save_button.click(save_chat, chatbot, None)
            #delete_button.click(delete_model, None, queue=False)
            
            #refresh_button.click(update_dropdown_list, inputs = [], outputs = [docs_dropdown])
            
            
            #file_loader.upload(update_doc_and_reload_QA_sys, file_loader).then(
            #    load_state_log, None, state_text)
           
        demo.queue()
        demo.launch(share=True)
        #demo.launch()
    
if __name__ == "__main__":
    QA_UI_CLS = QA_UI()
    QA_UI_CLS.QA_gradio_UI()   
                                                  