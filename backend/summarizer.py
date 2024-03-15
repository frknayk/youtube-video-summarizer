from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
import torch
from langchain import PromptTemplate,  LLMChain

model = "meta-llama/Llama-2-7b-chat-hf"
# model = "conceptofmind/Yarn-Llama-2-13b-128k"

tokenizer = AutoTokenizer.from_pretrained(model)
max_token_length = 1000

pipeline = transformers.pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=max_token_length,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

template = """
              Write a concise summary of the following text delimited by triple backquotes.
              Return your response in bullet points which covers the key points of the text.
              ```{text}```
              BULLET POINT SUMMARY:
           """
prompt = PromptTemplate(template=template, input_variables=["text"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

def summarize_video_transcript(transcript: str, summary_length: int = -1):
    summary = llm_chain.run(transcript)
    print(summary)
    # summary_all = []
    # for i in range(0, len(transcript), max_token_length):
    #     print("Index: ",i)
    #     summary = llm_chain.run(transcript[i:i+max_token_length])
    #     summary_all.append(summary)
    # return "\n".join(summary_all)

