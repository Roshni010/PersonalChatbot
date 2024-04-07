from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
output_parser = StrOutputParser()
#suruma prompt lekheko ani yo prompt uta runnable map ma chain lai gariaxa ani eha sort of fine tuning 
template="""
**Translation Task**

*Instructions:*
PLEASE TRANSLATE THE FOLLOWING QUESTION INTO ENGLISH IF IT IS NOT IN THE ENGLISH SCRIPT.

Strictly use the following Word-translation pair to optimize all translation tasks:
- kati=> how much
- k/kun/k ho/kun ho => what?
- kaha=>where?
- usle/usko/unko=>roshani
- horw=>oh okay!
- ohoo=>wow!
-leko=>take
-baba=>father
-mummy=>mother
-hajur buwa=>grandfather
-man parne=>favourite
-naam =>name
-didi=>sister
-bhai=>brother
-timro/temro/tmro/tero/roshani=>your
*Question:*
{question}

"""

prompt = ChatPromptTemplate.from_template(template)
# Initialize the generative AI model
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}
llmG = ChatGoogleGenerativeAI(model='gemini-1.0-pro', generation_config=generation_config)

chain = RunnableMap({
    "question": lambda x: x['question']
}) | prompt | llmG | output_parser

def translatedQuestion(query:str):
    result = chain.invoke({'question':query})
    print(result)
    return {"result":result}

