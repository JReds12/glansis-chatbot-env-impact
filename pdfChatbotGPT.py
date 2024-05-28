#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# IMPORT LIBRARIES
from personal import secrets
import os
import streamlit as st
import pandas as pd
import openai
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# OPENAI KEY 
api_key = secrets.get('OPENAI API KEY')
os.environ["OPENAI_API_KEY"] = str(api_key)


# CREATE ENVIRONMENTAL IMPACT ANALYSIS

# Description of environmental impacts
env_impact = f"""
Disease/Parasite/Toxicity: The species pose some hazard or threat to the health of native species (e.g., it magnifies toxin levels; is poisonous; is a pathogen, parasite, or a vector of either); the species has introduced a novel or rare disease or parasite to another organism in the area that was unafflicted with said disease or parasite before its introduction, including moving a native parasite outside of its typical range; toxicity includes both envenomation and poisoning. The species pose some hazard or threat to human health (e.g., it magnifies toxin levels, is poisonous, a virus, bacteria, parasite, or a vector of one).

Predation/Herbivory: species consumes or is consumed by another species. 
it alter predator-prey relationships.

Food Web: species changes second order or higher nutrient/feeding cascades.

Competition: The species out-compete native species for available resources (e.g., habitat, food, nutrients, light). species shares a niche with another species where introduced, such that they compete for resources (such as food and habitat).

Genetic: Has it affected any native populations genetically (e.g., through hybridization, selective pressure, introgression). Species hybridizes with another organism as a result of its introduction, with the resulting offspring viability being irrelevant.

Water Quality: species creates measurable changes in water chemistry/quality/parameters as compared to pre-introduction. negatively affect water quality (e.g., increased turbidity or clarity, altered nutrient, oxygen, or other chemical levels/cycles).

Habitat Alteration: introduction of the species modifies the environment to which it was introduced, such as zebra mussels that attached to surfaces, changing the substrate of a waterbody. 

alter physical components of the ecosystem in some way (e.g., facilitated erosion/siltation, altered hydrology, altered macrophyte/phytoplankton communities, physical or chemical changes to substrate)?
"""

# Description of study types
study_type = f"""
Experimental: a study/reference with a claim that was supported experimentally, i.e. at least one variable in the study was manipulated.

Observational: a study/reference that with a claim that was founded observing something, i.e. nothing in the study or report was a result of manipulating any variables.

Anecdotal: a study/reference with a claim that is unfounded with direct research, but supported by theory or correlation, therefore anecdotal.
"""

# Description of study locations
study_location = f"""
Field: The study/impact occurred in the field.

Laboratory: The study/impact occurred in the laboratory.

N/A: Study/impact was not in a lab or field setting and falls in neither of the previous categories
 """


def analyze_impacts(species):
    # Categorizing different impacts
    query = f"""
    What are the documented categorical ecological impacts of invasive ```{species}``` in invaded 
    regions? Generate a list separated by commas containing only “Disease/Parasite/Toxicity”, “Predation/Herbivory”, 
    “Food Web”, “Competition”, “Genetic”, “Water Quality”, or “Habitat Alteration” using
    ```{env_impact}``` as guidance.
    If there are no impacts report as NA
    """
    docs = docsearch.similarity_search(query)
    impact_types = chain.run(input_documents=docs, question=query).split(", ")

    # Study type
    study_type = []
    for impact in impact_types:
        query = f"""
        How was the impact '''{impact}''' documented? Use ```{study_type}``` as guidance for determining study type. 
        Possible responses are: “Experimental”, “Observational”, or “Anecdotal.” Do not add any additional content, 
        formatting, or punctuation.
        """
        docs = docsearch.similarity_search(query)
        study_type.append(chain.run(input_documents=docs, question=query))

    # Study location
    study_location = []
    for impact in impact_types:
        query = f"""
        Where did the documented impact '''{impact}''' occur? Use ```{study_location}``` as guidance for study location. 
        Possible responses are: “Field”, “Laboratory”, or “N/A.” Do not add any additional content or formatting.
        """
        docs = docsearch.similarity_search(query)
        study_location.append(chain.run(input_documents=docs, question=query))

    # Describe the impact
    impact_description = []
    for impact in impact_types:
        query = f"""
        Write one to three sentence descriptions of the '''{impact}''' of ```{species}``` on the
        environment. This should include the scientific name of the species, as well as the geographic
        location of the impact (Country/state, waterbody). Descriptions must be fewer than 500 characters.
        """
        docs = docsearch.similarity_search(query)
        impact_description.append(chain.run(input_documents=docs, question=query))

    # Experiment location
    geo_loc = []
    for impact in impact_types:
        query = f"""
        Where is the geographic location of the impact '''{impact}''' ? Report answer in the format 
        “waterbody, State/province, Country” or “waterbody, Country.” Do not add any additional content 
        or formatting. Do not use abbreviations for location, waterbody, or country.
        """
        docs = docsearch.similarity_search(query)
        geo_loc.append(chain.run(input_documents=docs, question=query))

    # Great Lakes - experiment location
    gl_loc = []
    for impact in impact_types:
        query = f"""
        Did this impact happen within the Great Lakes Basin? Possible responses are: “yes” or “no.”
        Do not add any additional content, formatting, or punctuation.
        """
        docs = docsearch.similarity_search(query)
        gl_loc.append(chain.run(input_documents=docs, question=query))

    # Impacted species
    impacted_sp = []
    for impact in impact_types:
        query = f"""
        If applicable, which native species were impacted by '''{species}''' from '''{impact}'''? Report as list of common 
        or scientific names. If none, report "NA." Do not add any additional content, formatting, or punctuation.
        """
        docs = docsearch.similarity_search(query)
        impacted_sp.append(chain.run(input_documents=docs, question=query))

    data = {
        "Impact Type": impact_types,
        "Study Type": study_type,
        "Study Location": study_location,
        "Impact Description": impact_description,
        "Geographic Location": geo_loc,
        "Great Lakes Region": gl_loc,
        "Impacted Species": impacted_sp
    }

    df = pd.DataFrame(data)
    return df


# INITIALIZE CONVERSATION

# Initialize session state to store conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

if "messages" not in st.session_state:
    st.session_state.messages = [] 

    
# WEB PAGE SETUP

# set up main page
st.title("GLANSIS PDF Chatbot")
st.write("This chatbot interface will allow GLANSIS users to ask PDFs a series of questions.")

# set up sidebar
st.sidebar.title("Get Invasive Species Environmental Impacts")

pdf = st.sidebar.file_uploader("Select PDF")

# code to process PDF
if pdf is not None:
    try: 
        # Read in text from PDF 
        raw_text = ''
        pdf_doc = PdfReader(pdf)
        for page in pdf_doc.pages:
            raw_text += page.extract_text()
            
    except:
        st.error("Please add PDF")
        
    # Split the text into smaller chunks
    text_splitter = CharacterTextSplitter(
        separator = '\n',
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
    )
    
    text_chunk = text_splitter.split_text(raw_text)
    
    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings()

    # Create vector database
    docsearch = FAISS.from_texts(text_chunk, embedding=embeddings)
    st.session_state["docsearch"] = docsearch
    
    # Question requirements
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type = 'stuff')
        
    # Display success message in chat interface
    st.success("PDF successfully processed! You can start asking questions.")

species_name = st.sidebar.text_input("Enter Species Names")
if st.sidebar.button('Run'):
    df = analyze_impacts(species_name)
    st.sidebar.markdown('----')
    csv = df.to_csv()
    st.sidebar.download_button(
        label='Download CSV',
        data=csv,
        file_name='impacts.csv',
        mime='text/csv'
)

# CREATE MESSAGE BOT
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
if prompt := st.chat_input("What are your questions?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)  
        
    docsearch = st.session_state.get("docsearch")
    
    if docsearch:
        docs = docsearch.similarity_search(prompt)
        response = chain.run(input_documents=docs, question=prompt)
    else:
        response = "Please upload a PDF first."
        
    with st.chat_message("assistant"):
        st.markdown(response)
        
    st.session_state.messages.append({"role": "assistant", "content": response})

