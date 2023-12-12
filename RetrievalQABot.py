import gradio as gr
import os
import time

from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from huggingface_hub import notebook_login
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
import textwrap
import sys
import torch
import json

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class QA_LangChain():
    def __init__(self):
        self.llm = None
    
    def QA_LangChain_RQA_chain(self, input_doc_pth):
        # load embedding model
        print("===== Load the embedding model =====")
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device': 'cpu'})
        vectorstore=FAISS.load_local(input_doc_pth, embeddings)
        
        # Create a QA chain
        print("===== Create a QA chain =====")    
        chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", return_source_documents=True, retriever=vectorstore.as_retriever())
        
        return chain
    
    def QA_LangChain_RQA_model_create(self, llm_pth):
        
        # Load LLM model
        print("===== Load LLM model =====")
        tokenizer = AutoTokenizer.from_pretrained(llm_pth)
        model = AutoModelForCausalLM.from_pretrained(llm_pth, device_map='auto',
                           torch_dtype=torch.float16,
                           use_auth_token=True,
                           #load_in_8bit=True,
                            #load_in_4bit=True
                                                    )
        pipe = pipeline("text-generation",
             model=model,
             tokenizer= tokenizer,
             torch_dtype=torch.bfloat16,
             device_map="auto",
             max_new_tokens = 2048,
             do_sample=True,
             top_k=10,
             num_return_sequences=1,
             eos_token_id=tokenizer.eos_token_id
                       )
        
        # Create LLM model                                              
        self.llm=HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':0})

    def QA_LangChain_RQA_model_CodeLlama_34B_create(self, llm_pth):
        
        # Load LLM model
        print("===== Load LLM model =====")
        quantization_config = BitsAndBytesConfig(
           load_in_4bit=True,
           bnb_4bit_compute_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(llm_pth)
        model = AutoModelForCausalLM.from_pretrained(
            llm_pth, device_map="auto",
            quantization_config=quantization_config,
        )

        # code llama
        pipe = pipeline("text-generation",
             #"question-answering",
             model=model,
             tokenizer= tokenizer,
             #torch_dtype=torch.bfloat16,
             torch_dtype=torch.float16,
             device_map="auto",
             max_new_tokens = 2048,
             #max_length=500,
             do_sample=True,
             top_k=10,
             top_p=0.9,
             temperature=0.1,
             num_return_sequences=1,
             eos_token_id=tokenizer.eos_token_id
             )
        
        # Create LLM model                                              
        self.llm=HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':0})


class QA_UI():
    def __init__(self):
        self.qa_doc_dir = r'D:\nu_QA_data'

        #self.llm_model_path = r'C:\Users\USER\Desktop\llma\text-generation-webui\models\Llama-2-7b-chat-hf'
        #self.llm_model_path = r'meta-llama/Llama-2-13b-chat-hf'
        #self.llm_model_path = r'C:\Users\USER\Desktop\llma\text-generation-webui\models\stabilityai_StableBeluga-7B'
        #self.llm_model_path = r'C:\Users\USER\Desktop\llma\text-generation-webui\models\Llama-2-7b-chat-hf'
        #self.llm_model_path = r'D:\replit-code-v1_5-3b'
        #self.llm_model_path = r'D:\CodeLlama-7b-Instruct-hf'
        self.llm_model_path = r'C:\Users\USER\Desktop\llma\text-generation-webui\models\CodeLlama-34b-Instruct-hf'
        
        self.qa_chain = None
        #self.doc_path = r'D:\nu_QA_data\m460bsp_SampleCode_StdDriver_wo_headers'
        self.doc_path = r'D:\nu_QA_data\m460bsp_Library_StdDriver_headers_1000'
        #self.doc_path = r'D:\nu_QA_data\m460bsp_Library_StdDriver'
        
        # We need to initial the llm model when intial the class & can't create another
        self.QA_LangChain_CLS = None
        self.init_chat_qa_bot()

    def init_chat_qa_bot(self):
        self.QA_LangChain_CLS = QA_LangChain()
        #self.QA_LangChain_CLS.QA_LangChain_RQA_model_create(self.llm_model_path)
        # 11/15 CODE-LLAMA VERSION
        self.QA_LangChain_CLS.QA_LangChain_RQA_model_CodeLlama_34B_create(self.llm_model_path)
        self.qa_chain = self.QA_LangChain_CLS.QA_LangChain_RQA_chain(self.doc_path)    

    def QA_gradio_UI(self):
        
        def start_qa_chain(doc_name):
            self.doc_path = os.path.join(self.qa_doc_dir, doc_name)
            #print(self.doc_path)
            #time.sleep(2)
            self.qa_chain = self.QA_LangChain_CLS.QA_LangChain_RQA_chain(self.doc_path)
  
            return "{} finish !".format(doc_name.split('_index')[0])
        
        inputs = [gr.Dropdown(next(os.walk(self.qa_doc_dir))[1], label="Choose Documents", info=self.qa_doc_dir, value=self.doc_path.split('\\')[-1]),
                  ]
        outputs = [gr.Textbox(label='State of loading document & model', value=self.doc_path.split('\\')[-1])
            ]
        
        with gr.Blocks() as demo:
            
            # Layout Section
            with gr.Tab("QA ChatBot"):
                gr.Markdown(
                    value= "LLM model: {}".format(self.llm_model_path.split("\\")[-1]))
                with gr.Row():
                    def update_dropdown_list():
                        return [gr.Dropdown(choices=next(os.walk(self.qa_doc_dir))[1]), 'loading...']
                    
                    gr.Interface(fn=start_qa_chain, inputs=inputs, outputs=outputs)
                    # if dropdown is been selected, it will refresh 1 time
                    inputs[0].select(fn=update_dropdown_list, inputs=[], outputs=[inputs[0], outputs[0]])
                    
                chatbot = gr.Chatbot(height = 600, show_copy_button = True, scale = 2)
                msg = gr.Textbox()
                with gr.Row():
                    enter_button = gr.Button("Enter")
                    save_button = gr.Button("Save")
                    clear = gr.Button("Clear")
                    #delete_button = gr.Button("delete model")
            
            with gr.Tab("Question Example"):
                gr.Markdown(
                        """
                        # Nuvoton-MCU-PSG
                        #### What are the key features of M460 Series?
                        #### What is the operating voltage of M031G?
                        #### What is the architecture of NuMicro M23?
                        """
                )
            
            with gr.Tab("Documents (ToDo...)"):
                file_loader = gr.File(file_count="multiple", file_types=[".txt", ".pdf", ".csv"], height=100)
                state_text = gr.Textbox(label='model state:', value="Please load a document.")
            
            
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
                result= self.qa_chain({"query": query}, return_only_outputs=True)
                #bot_message = textwrap.fill(result['result'], width=500)
                bot_message = result['result']
                                                         
                history[-1][1] = ""

                history[-1][1] = bot_message
                #print(history[-1][1])
                return history

                #for character in bot_message:
                #    history[-1][1] += character
                #    time.sleep(0.05)
                #    yield history
                   
            def save_chat(history):
                with open('{}.json'.format(time.strftime("%Y%m%d-%H%M%S")), 'w', encoding='utf-8') as f:
                    json.dump(history, f, ensure_ascii=False, indent=4)
                    
            #def delete_model():
            #    del self.qa_chain
        
            # Interactive Section
            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, chatbot, chatbot)
            enter_button.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, chatbot, chatbot)
            clear.click(lambda: None, None, chatbot, queue=False)
            save_button.click(save_chat, chatbot, None)
            #delete_button.click(delete_model, None, queue=False)
            
            #refresh_button.click(update_dropdown_list, inputs = [], outputs = [docs_dropdown])
            
            
            file_loader.upload(update_doc_and_reload_QA_sys, file_loader).then(
                load_state_log, None, state_text)
           
        demo.queue()
        demo.launch(share=True)
        #demo.launch()
    
if __name__ == "__main__":
    QA_UI_CLS = QA_UI()
    QA_UI_CLS.QA_gradio_UI()   
                                                  