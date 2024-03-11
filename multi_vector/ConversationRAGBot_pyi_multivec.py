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
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

#from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings

from langchain.callbacks import get_openai_callback

from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.docstore.document import Document
import pickle

import APIKEY

#import lancedb
#from langchain.vectorstores import LanceDB

# SSL Issue
os.environ['REQUESTS_CA_BUNDLE'] = "cacert.pem"

class QA_LangChain():
    def __init__(self):
        self.llm = None
        self.retriever_vec = None
        self.memory = None
        #self.gptmodel = "gpt-3.5-turbo-1106" #"gpt-3.5-turbo"
        #self.gptmodel = "gpt-4"
        self.gptmodel = "gpt-4-1106-preview"
        self.mmr_num = 8


        
#        self.system_template = """Answer the question based only on the following NAU8822A CODEC context and chat history, 
#which can include text and tables:
#----------------
#{context}
#{chat_history}"""

        self.system_template = """Use the following pieces of context and chat history to answer the question at the end. The context is Nuvoton M467 Series Technical Reference Manual.
If you don't know the answer or the question has nothing to do with code or programing, don't try to make up an answer.
----------------
{context}
{chat_history}"""

        
        messages = [
        SystemMessagePromptTemplate.from_template(self.system_template),
        HumanMessagePromptTemplate.from_template("{question}")
        ]
        self.qa_prompt = ChatPromptTemplate.from_messages(messages)
        
    def Create_Chain(self, qa_prompt):

        return ConversationalRetrievalChain.from_llm(llm=self.llm, retriever=self.retriever_vec, memory=self.memory,
                                                      return_source_documents=True, 
                                                      combine_docs_chain_kwargs={"prompt": qa_prompt}
                                                     )

    def QA_LangChain_RQA_chain(self, input_doc_pth, qa_prompt):
        # load embedding model
        print("===== Load the embedding model =====")
        
        # choose your embeddings model
        #embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device': 'cpu'})
        #embeddings = HuggingFaceInstructEmbeddings(model_name='BAAI/bge-small-en',model_kwargs={'device': 'cpu'})
        embeddings = OpenAIEmbeddings()

        DOC_SAVE_NAME = input_doc_pth

        # Load summaries pickle
        SUM_list = []
        with open(DOC_SAVE_NAME + r".pickle", "rb") as f:
            SUM_list = pickle.load(f)
        texts_list = SUM_list[0]
        tables_list = SUM_list[1]
        print("The raw texts chunks: {} The raw tables chunk: {}".format(len(texts_list), len(tables_list)))

        text_list_update = []
        for info in texts_list:
            doc =  Document(page_content=info[1], metadata={"source": "texts from {}".format(str(Path(DOC_SAVE_NAME).stem))})
            text_list_update.append([info[0], doc])
        table_list_update = []
        for info in tables_list:
            doc =  Document(page_content=info[1], metadata={"source": "texts from {}".format(str(Path(DOC_SAVE_NAME).stem))})
            table_list_update.append([info[0], doc])
        
        # Debug, check the Document obj
        #print(type(text_list_update[0]))
        #print(text_list_update[0])

        # FAISS  
        vectorstore=FAISS.load_local(input_doc_pth, embeddings)

        # The retriever (empty to start)
        store = InMemoryStore()
        id_key = "doc_id"
        self.retriever_vec = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
        )
        self.retriever_vec.docstore.mset(text_list_update)
        self.retriever_vec.docstore.mset(table_list_update)
     
        print("===== Create a ConversationalRetrievalChain chain =====")
        # Normal memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", input_key='question', output_key='answer', return_messages=True)
        # Should save the tokens
        #memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
        
        chain = self.Create_Chain(qa_prompt)
        
        return chain
    
    def QA_LangChain_RQA_model_Chatgpt_create(self):
        os.environ["OPENAI_API_KEY"] = APIKEY.API_KEY_SERVICE_OPENAI
        self.llm = ChatOpenAI(temperature=0, model=self.gptmodel)


class QA_UI():
    def __init__(self):
        self.qa_doc_dir = r'doc_faiss_multi'
        #self.qa_doc_dir = r'lanceDB'

        #self.llm_model_path = r'C:\Users\USER\Desktop\llma\text-generation-webui\models\Llama-2-7b-chat-hf'
        #self.llm_model_path = r'meta-llama/Llama-2-13b-chat-hf'
        #self.llm_model_path = r'C:\Users\USER\Desktop\llma\text-generation-webui\models\stabilityai_StableBeluga-7B'
        #self.llm_model_path = r'C:\Users\USER\Desktop\llma\text-generation-webui\models\Llama-2-7b-chat-hf'
        #self.llm_model_path = r'D:\replit-code-v1_5-3b'
        #self.llm_model_path = r'D:\CodeLlama-7b-Instruct-hf'
        #self.llm_model_path = r'C:\Users\USER\Desktop\llma\text-generation-webui\models\CodeLlama-34b-Instruct-hf'
        
        self.qa_chain = None
        #self.doc_path = os.path.join(self.qa_doc_dir, r'DS_NAU8822A_DataSheet_3.5')
        self.doc_path = os.path.join(self.qa_doc_dir, r'TRM_M463_M467')
    
        
        # We need to initial the llm model when intial the class & can't create another
        self.QA_LangChain_CLS = None
        self.init_chat_qa_bot()

        # save the RQA's docs
        self.ref_docs = None
        self.return_msg = ""

        # save cb info for tokens
        self.cb_info = None
        self.tokens_str = ""

        # save the RQA's docs detail
        self.return_msg_detail = ""

    def init_chat_qa_bot(self):
        self.QA_LangChain_CLS = QA_LangChain()
        #self.QA_LangChain_CLS.QA_LangChain_RQA_model_create(self.llm_model_path)
        # 11/15 CODE-LLAMA VERSION
        #self.QA_LangChain_CLS.QA_LangChain_RQA_model_CodeLlama_34B_create(self.llm_model_path)
        # 11/30 Chat-GPT 3.5 turbo
        self.QA_LangChain_CLS.QA_LangChain_RQA_model_Chatgpt_create()
        self.qa_chain = self.QA_LangChain_CLS.QA_LangChain_RQA_chain(self.doc_path, self.QA_LangChain_CLS.qa_prompt)
        #print("ConversationalRetrievalChain detail:")
        #print(self.qa_chain)


    def QA_gradio_UI(self):
        
        def start_qa_chain(doc_name):
            self.doc_path = os.path.join(self.qa_doc_dir, doc_name)
            #print(self.doc_path)
            #time.sleep(2)
            self.qa_chain = self.QA_LangChain_CLS.QA_LangChain_RQA_chain(self.doc_path,  self.QA_LangChain_CLS.qa_prompt)
            print("Re-create => ConversationalRetrievalChain detail:")
            print(self.qa_chain)
  
            return "{} finish !".format(doc_name.split('_index')[0])
        
        inputs = [gr.Dropdown(next(os.walk(self.qa_doc_dir))[1], label="Choose Documents", info=self.qa_doc_dir, value=self.doc_path.split('\\')[-1]),
                  ]
        outputs = [gr.Textbox(label='State of loading document', value=self.doc_path.split('\\')[-1])
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
        
        def return_ref_docs_detail(): 
           for i in range(len(self.ref_docs)):
               if 'page' in self.ref_docs[i].metadata:
                   self.return_msg_detail += "\n" + str(i)+": " + "\n" + self.ref_docs[i].metadata['source'] + ":" + str((self.ref_docs[i].metadata['page']+1)) + " " + self.ref_docs[i].page_content
               else: 
                   self.return_msg_detail += "\n" + str(i)+": " + "\n" + self.ref_docs[i].metadata['source'] + " " + self.ref_docs[i].page_content
           self.return_msg_detail += "\n" + "---------------------"         
           return self.return_msg_detail
        
        def return_ref_docs_detail_clear():
                self.return_msg_detail = ""
                return self.return_msg_detail
         
        with gr.Blocks() as demo:    
            # Layout Section
            with gr.Tab("QA ChatBot"):
                gr.Markdown(value = "MultiVectorRetriever Version")
                gr.Markdown(
                    #value= "LLM model: {}".format(self.llm_model_path.split("\\")[-1]))
                    value= "LLM model: {}".format(self.QA_LangChain_CLS.gptmodel))
                with gr.Row():
                    def update_dropdown_list():
                        return [gr.Dropdown(choices=next(os.walk(self.qa_doc_dir))[1]), 'loading...']
                    
                    gr.Interface(fn=start_qa_chain, inputs=inputs, outputs=outputs)
                    # if dropdown is been selected, it will refresh 1 time
                    inputs[0].select(fn=update_dropdown_list, inputs=[], outputs=[inputs[0], outputs[0]])
                    
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
            
            with gr.Tab("Return Source Detail"):
                 ref_doc_detail = gr.Textbox(label="Reference Documents' Page Content",
                                     info="these are raw chunk form documents",
                                     lines=15,
                                     value=" ")

            with gr.Tab("System Message Prompt"):
                gr.Markdown(
                        """
                        # Nuvoton-MCU QA and CodeGen Prompt
                        #### Please update the LLM prompt template basing on your different task. For example:
                        #### QA: 
                        
                        ```
                        Use the following pieces of context and chat history to answer the question at the end. The context is Nuvoton M467 Series Technical Reference Manual.
                        If you don't know the answer or the question has nothing to do with code or programing, don't try to make up an answer.
                        ----------------
                        {context}
                        {chat_history}
                        ```

                        #### CodeGen: 
                        ```
                        Use the following pieces of context and chat history to answer the question at the end. The context is standard driver C header files of M460 MCU.
                        Please answer with C Code function as complete as possible.
                        If you don't know the answer or the question has nothing to do with code or programing, don't try to make up an answer.
                        ----------------
                        {context}
                        {chat_history}
                        ```
                        """
                )
                prompt_inp = gr.Textbox(label="Prompt Edit", lines=4)
                prompt_oup = gr.Textbox(label="Prompt Now", lines=4, placeholder=self.QA_LangChain_CLS.system_template)
                up_prompt_btn = gr.Button("Update Prompt")

                #@up_prompt_btn.click(inputs=prompt_inp, outputs=prompt_oup)
                def update_prompt(prompt_inp):
                    
                    update_messages = [
                    SystemMessagePromptTemplate.from_template(prompt_inp),
                    HumanMessagePromptTemplate.from_template("{question}")
                    ]
                    self.QA_LangChain_CLS.qa_prompt = ChatPromptTemplate.from_messages(update_messages)
                    
                    # re-create llm chain with new prompt
                    self.qa_chain = self.QA_LangChain_CLS.QA_LangChain_RQA_chain(self.doc_path,  self.QA_LangChain_CLS.qa_prompt)
                    print("Re-create => ConversationalRetrievalChain detail:")
                    print(self.qa_chain)

                    return prompt_inp
                
                # update the prompt, so need to 1. rebuild the RQAchain, 2. clear the chatbox & info
                up_prompt_btn.click(fn=update_prompt, inputs=prompt_inp, outputs=prompt_oup).then(
                    lambda: None, None, chatbot, queue=False).then(
                        return_ref_docs_clear, None, ref_doc).then(return_tokens_clear, None, cb_tok_box).then(return_ref_docs_detail_clear, None, ref_doc_detail)
            

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
                    result= self.qa_chain({"question": query})
                    #bot_message = textwrap.fill(result['result'], width=500)
                    #print(result)
                    
                    bot_message = result['answer']                                      
                    history[-1][1] = ""
                    history[-1][1] = bot_message
                    #print(history[-1][1])
    
                    self.ref_docs = result['source_documents']
                    #for i in range(len(self.ref_docs)):
                    #    print(self.ref_docs[i].metadata['source'])

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
                #self.qa_chain =  self.QA_LangChain_CLS.Create_Chain()

                # 11/30 only need to clear the memory
                #print(self.qa_chain.memory)
                self.qa_chain.memory.clear()
                print("create new chat:")
                print(self.qa_chain.memory)
                
            # Interactive Section
            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, chatbot, chatbot).then(return_ref_docs, None, ref_doc).then(return_tokens, None, cb_tok_box).then(return_ref_docs_detail, None, ref_doc_detail)
            
            enter_button.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, chatbot, chatbot).then(return_ref_docs, None, ref_doc).then(return_tokens, None, cb_tok_box).then(return_ref_docs_detail, None, ref_doc_detail)
            
            clear.click(lambda: None, None, chatbot, queue=False).then(
                create_new_chain, None, None).then(return_ref_docs_clear, None, ref_doc).then(return_tokens_clear, None, cb_tok_box).then(return_ref_docs_detail_clear, None, ref_doc_detail)
            
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
                                                  