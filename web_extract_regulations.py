from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_openai import ChatOpenAI
from langchain.chains import create_extraction_chain
import pprint
from langchain_text_splitters import RecursiveCharacterTextSplitter

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

schema = {
    "properties": {
        "law_content": {"type": "string", "description": "The content of the law"},
    },
}

def extract(content: str, schema: dict):
    return create_extraction_chain(schema=schema, llm=llm).invoke(content)

def scrape_with_playwright(urls, schema):
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        docs, tags_to_extract=["span"]
    )
    print("Extracting content with LLM")
    print(f"{docs_transformed[0].page_content=}")
    extracted_content = extract(schema=schema, content=docs_transformed[0].page_content)
    # pprint.pprint(extracted_content)
    return extracted_content

    # # Grab the first 1000 tokens of the site
    # splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    #     chunk_size=1000, chunk_overlap=0
    # )
    # splits = splitter.split_documents(docs_transformed)

    # # Process the first split
    # extracted_content = extract(schema=schema, content=splits[0].page_content)
    # pprint.pprint(extracted_content)
    # return extracted_content


urls = ["https://law.moea.gov.tw/LawContent.aspx?id=FL068073#lawmenu"]
extracted_content = scrape_with_playwright(urls, schema=schema)
