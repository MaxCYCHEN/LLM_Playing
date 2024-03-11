import os
import time
from pathlib import Path
import traceback
import json
import sys
import argparse

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

from langsmith import Client

#import lancedb
#from langchain.vectorstores import LanceDB

# SSL Issue
#os.environ['REQUESTS_CA_BUNDLE'] = "cacert.pem"

class QA_LangChain():
    def __init__(self, chip_type, app_type, llm_model):
        self.apikey_file_pth = 'APIKEY.txt'
        self.app_type = app_type
        self.llm = None
        self.retriever_vec = None
        self.memory = None
        #self.gptmodel = "gpt-3.5-turbo-1106"
        #self.gptmodel = "gpt-4"
        #self.gptmodel = "gpt-4-1106-preview"
        self.gptmodel = llm_model
        self.mmr_num = 8

        try:
            f = open("sys_prompt.json", 'r')
            prompt_data = json.load(f)
        except Exception as e:
            print(e, file=sys.stderr, flush=True)    

        if self.app_type.upper() == "QA":
            self.system_template = prompt_data['QA'].replace(r'{chip_type}', chip_type)
#            self.system_template = """Use the following pieces of context and chat history to answer the question at the end. The context is Nuvoton """+chip_type+""" Series Technical Reference Manual.
#If you don't know the answer or the question has nothing to do with technical, don't try to make up an answer.
#----------------
#{context}
#{chat_history}"""
        elif self.app_type.upper() == "CODEGEN":
            self.system_template = prompt_data['CODEGEN'].replace(r'{chip_type}', chip_type)
#            self.system_template = """Use the following pieces of context and chat history to answer the question at the end. The context is standard driver C header files of """+chip_type+""" MCU.
#Please answer with C Code function as complete as possible.
#If you don't know the answer or the question has nothing to do with code or programing, don't try to make up an answer.
#----------------
#{context}
#{chat_history}"""
        else:
            self.system_template = """Use the following pieces of context and chat history to answer the question at the end. The context are """+chip_type+""" MCU standard driver C header files and technical reference manual.
If the question is about generating code, please answer with C Code function from the context as complete as possible.
Otherwise, answer the M2354 MCU relative question with the context of technical reference manual.
If you don't know the answer or the question has nothing to do with code or programing, don't try to make up an answer.
----------------
{context}
{chat_history}"""
        
        messages = [
        SystemMessagePromptTemplate.from_template(self.system_template),
        HumanMessagePromptTemplate.from_template("{question}")
        ]
        self.qa_prompt = ChatPromptTemplate.from_messages(messages)
        
    def Create_Chain(self):

        #print(self.qa_prompt, flush=True) # debug the prompt is ok or not
        return ConversationalRetrievalChain.from_llm(llm=self.llm, retriever=self.retriever_vec, memory=self.memory,
                                                      return_source_documents=True, 
                                                      combine_docs_chain_kwargs={"prompt": self.qa_prompt}
                                                     )

    def QA_LangChain_RQA_chain(self, input_doc_pth):
        # load embedding model
        print("===== Load the embedding model =====", flush=True)
        
        # choose your embeddings model
        embeddings = OpenAIEmbeddings()
 
        # FAISS  
        vectorstore=FAISS.load_local(input_doc_pth, embeddings)

        self.retriever_vec=vectorstore.as_retriever(
        search_type="mmr", # Also test "similarity"
        search_kwargs={"k": self.mmr_num})
     
        print("===== Create a ConversationalRetrievalChain chain =====", flush=True)
        # Normal memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", input_key='question', output_key='answer', return_messages=True)
        
        chain = self.Create_Chain()
        
        return chain
    
    def QA_LangChain_RQA_model_Chatgpt_create(self):
        API_line = ""
        smith_API_line = ""
        try:
            file1 = open(self.apikey_file_pth, "r")
            API_line = file1.readline()
            smith_API_line = file1.readline()
            file1.close()
        except Exception as e:
            print(e, file=sys.stderr, flush=True)

        os.environ["OPENAI_API_KEY"] = API_line.strip()
        os.environ["LANGCHAIN_API_KEY"] = smith_API_line.strip()

        try:             
            self.llm = ChatOpenAI(temperature=0, model=self.gptmodel)
        except Exception as e:
            print(e, file=sys.stderr, flush=True)
            

class FileModified():
    def __init__(self, file_path, callback):
        self.file_path = file_path
        self.callback = callback
        self.modifiedOn = os.path.getmtime(file_path)

    def start(self):
        try:
            time_count = 0
            while (True):
                time.sleep(1)
                modified = os.path.getmtime(self.file_path)
                time_count+=1
                print(time_count, flush=True)
                if modified != self.modifiedOn:
                    self.modifiedOn = modified
                    if self.callback():
                        break
        except Exception as e:
            print(traceback.format_exc(), file=sys.stderr, flush=True)            

class main_loop():
    
    def __init__(self, app_type, llm_model):
        self.app_type = app_type
        self.llm_model = llm_model
        self.qa_doc_dir = os.path.join(r'doc_faiss', r'QA')
        self.codegen_doc_dir = os.path.join(r'doc_faiss', r'CODEGEN')
        self.qa_chain = None
        self.doc_path = ""
        
        # We need to initial the llm model when intial the class & can't create another
        self.QA_LangChain_CLS = None

        # save the RQA's docs
        self.ref_docs = None
        self.return_msg = ""

        # save cb info for tokens
        self.cb_info = None
        self.tokens_str = ""

        self.input_fileName = r"input.json"
        self.output_fileName = r"output.json"
        self.special_split_key = r"[@NewJsonObjStart@]"
        self.chip_type = ""

        self.langsmith_en = 0

    def read_json(self, file_name):
        try:
            f = open(file_name, 'r')
            data = json.load(f)
        except Exception as e:
            print(e, file=sys.stderr, flush=True)    
        
        print("Read input json done!", flush=True)
        print("\n", flush=True) 
        return data
    
    def write_json(self, file_name, out_dict, ask_count):
        try:
            with open(file_name, "a") as outfile:
                if ask_count > 0:
                    outfile.write(self.special_split_key)
                json.dump(out_dict, outfile, indent = 4)
        except Exception as e:
            print(e, file=sys.stderr, flush=True)
        print("Write output json done!", flush=True)
        print("\n", flush=True)    

    def __return_ref_docs(self, ref_docs):
        ref_docs_list = []
        for i in range(len(ref_docs)):
            ele_dict = {}
            #ele_dict['pageContent'] = ref_docs[i].page_content # The content detail no need so far
            if 'page' in ref_docs[i].metadata:
                ele_dict['source'] = ref_docs[i].metadata['source']
                ele_dict['page'] = str((ref_docs[i].metadata['page']+1))
            else: 
                ele_dict['source'] = ref_docs[i].metadata['source']
            ref_docs_list.append(ele_dict)
        return ref_docs_list

    def __return_tokens(self, cb):
        cb_dict = {}
        cb_dict['Total'] = cb.total_tokens
        cb_dict['Prompt'] = cb.prompt_tokens
        cb_dict['Completion'] = cb.completion_tokens
        cb_dict['Cost(USD)'] = cb.total_cost   
        return cb_dict   
        
    def ask_chain(self, input_q, ask_count):
        print("Start to ask LLM, and please wait a while! ASK Count: {}".format(str(ask_count+1)), flush=True)
        print("\n", flush=True)
        
        with get_openai_callback() as cb:                                        
            result= self.qa_chain({"question": input_q})
            #print(result)
            bot_message = result['answer']                                      
            self.ref_docs = self.__return_ref_docs(result['source_documents'])
            self.cb_info = self.__return_tokens(cb)    
        return bot_message
    
    def initial_chain(self):
        # Create the QA Chain
        self.QA_LangChain_CLS = QA_LangChain(self.chip_type, self.app_type, self.llm_model)
        self.QA_LangChain_CLS.QA_LangChain_RQA_model_Chatgpt_create()
        self.qa_chain = self.QA_LangChain_CLS.QA_LangChain_RQA_chain(self.doc_path)

    def __chip_trm_select(self):

        if self.app_type.upper() == "QA":
            doc_path = None
            files_dir = [f for f in Path(self.qa_doc_dir).iterdir() if Path.is_dir(f)]
            for dir_pth in files_dir:
                #if re.search(r"M46*", (str)(Path(dir_pth).stem), flags = re.I):
                if (str)(Path(dir_pth).stem).find(self.chip_type.upper())!=-1:  
                    doc_path = dir_pth
                    #print(doc_path)
            assert doc_path!=None, "QA: The chip_type: {} not support or missing the faiss doc!".format(self.chip_type)
        elif self.app_type.upper() == "CODEGEN":
            doc_path = None
            files_dir = [f for f in Path(self.codegen_doc_dir).iterdir() if Path.is_dir(f)]
            for dir_pth in files_dir:
                #if re.search(r"M46*", (str)(Path(dir_pth).stem), flags = re.I):
                if (str)(Path(dir_pth).stem).find(self.chip_type.upper())!=-1:  
                    doc_path = dir_pth
                    #print(doc_path)
            assert doc_path!=None, "CODEGEN: The chip_type: {} not support or missing the faiss doc!".format(self.chip_type)
        else:
            print("Error, wrong app.type, no document load")    

        return doc_path

    def run_loop(self):

        def qinput_file_modified():
            print("\n", flush=True)
            print("Q Input File Modified!", flush=True)
            print("\n", flush=True)
            return True
  
        ask_count = 0
        while (True):
            In_fileModifiedHandler = FileModified(self.input_fileName, qinput_file_modified)
            print("\n", flush=True)
            print("Wait for question......", flush=True)
            In_fileModifiedHandler.start()

            # After the In_fileModifiedHandler.start() loop break, we get a new question
            input_data = self.read_json(self.input_fileName)

            # First time, cretae chain or user update the chip type 
            if ask_count==0 or (self.chip_type != input_data["chip_type"]):
                if ask_count!=0:
                    print("User update the chip_type!", flush=True)
                self.chip_type = input_data["chip_type"]
                assert self.chip_type!="", "Missing chip_type in input.json!" 
                self.doc_path = self.__chip_trm_select() 
                self.initial_chain()
                print(self.qa_chain, flush=True)

            # Check langsmith upload or not 
            if self.langsmith_en != input_data["langsmith"]:
                self.langsmith_en = input_data["langsmith"]
                if self.langsmith_en == 1:
                    os.environ["LANGCHAIN_TRACING_V2"] = "true"
                    os.environ["LANGCHAIN_PROJECT"] = "RAGchain_pyi_v0.0.6"
                    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
                    client = Client()
                    print("Enable the langsmith to upload traces!", flush=True)
                else:
                    os.environ["LANGCHAIN_TRACING_V2"] = "false"
                    print("Disable the langsmith!", flush=True)
               
   
                    
            assert input_data["question"]!=None, "Missing question in input.json!"
            bot_ans = self.ask_chain(input_data["question"], ask_count)
            assert bot_ans!=None, "The answer is empty!"

            #print(bot_ans)
            #print("\n")
            #print("Memory Check")
            #print(self.qa_chain.memory)
            #print("\n")
            #print(self.ref_docs)
            #print("\n")
            #print(self.cb_info)
            #break
            
            output_dict = {}
            output_qa_dict = {}
            output_qa_dict["app_type"] = self.app_type.upper()
            output_qa_dict["chip_type"] = self.chip_type
            output_qa_dict["question"] = input_data["question"]
            output_qa_dict["answer"] = bot_ans
            output_qa_dict["sourceDocuments"] = self.ref_docs
            output_qa_dict["tokens"] = self.cb_info
            output_dict[input_data["timestamp"]] = output_qa_dict
            self.write_json(self.output_fileName, output_dict, ask_count)

            ask_count+=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--app_type',
        choices=['QA', 'CODEGEN'],
        default='QA',
        help='Choose the LLM app type, we have QA and CodeGen.')
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-3.5-turbo-1106',
        help='Choose the chatGPT model, for example: gpt-3.5-turbo-1106, gpt-4-0125-preview.')
    args = parser.parse_args()

    QA_CLS = main_loop(args.app_type, args.model)
    QA_CLS.run_loop() 



    
  
                                                  