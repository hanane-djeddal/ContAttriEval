import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from tqdm import tqdm
import re
import logging
import os
import nltk 
import torch.nn.functional as F
import sys
import copy
from nltk import word_tokenize
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import random
import inflect


#from bert_score import score

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ['HF_HOME'] = os.environ['WORK'] + '/.cache/huggingface'
ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(ROOT_PATH)

from tools.citation_tools import remove_citations

import re
import random

#MAX_MODEL_LEN = 8192  #8192 #16384	

MAX_MODEL_LEN = 8192 
MAX_NEW_TOKENS = 1024 # You must set a value for your desired maximum output length

# Calculate the maximum allowed prompt length
MAX_PROMPT_FOR_TRUNCATION = MAX_MODEL_LEN - MAX_NEW_TOKENS

def create_wrong_value(match):
    # The matched string value (e.g., '1850', '75%', '10.5')
    original_str = match.group(0)

    # 1. Handle Years (Simple 4-digit numbers)
    if re.fullmatch(r'\d{4}', original_str):
        try:
            year = int(original_str)
            # Change by a random number between -25 and 25 (excluding 0)
            offset = random.choice([i for i in range(-25, 26) if i != 0])
            new_year = year + offset
            return str(new_year)
        except ValueError:
            pass # Keep original if conversion fails
    
    # 2. Handle Percentages
    elif original_str.endswith('%'):
        try:
            # Remove % and convert to float
            val = float(original_str.replace('%', '').strip())
            # Add a small random offset (e.g., -5 to 5)
            offset = random.uniform(-5, 5) 
            new_val = val + offset
            return f"{round(new_val, 2)}%" # Keep formatting
        except ValueError:
            pass

    # ... Add more logic for other types (decimals, dates, etc.) ...

    # Default: If no specific logic applies, return a slightly offset integer
    try:
        if '.' not in original_str and ',' not in original_str: # Simple integer
             val = int(original_str)
             offset = random.choice([-1, 1])
             return str(val + offset)
    except ValueError:
        pass

    # Return the original if no transformation was applied
    return original_str


def transcript_numbers(match):
    p = inflect.engine()
    # The matched string value (e.g., '1850', '75%', '10.5')
    original_str = match.group(0)

    # 1. Handle Years (Simple 4-digit numbers)
    if re.fullmatch(r'\d{4}', original_str):
        try:
            year = int(original_str)
            new_year =p.number_to_words(year)
            return str(new_year)
        except ValueError:
            pass # Keep original if conversion fails
    
    # 2. Handle Percentages
    elif original_str.endswith('%'):
        try:
            # Remove % and convert to float
            val = float(original_str.replace('%', '').strip())
            new_val = p.number_to_words(val)
            return f"{new_val}%" # Keep formatting
        except ValueError:
            pass

    # ... Add more logic for other types (decimals, dates, etc.) ...

    # Default: If no specific logic applies, return a slightly offset integer
    try:
        if '.' not in original_str and ',' not in original_str: # Simple integer
             val = int(original_str)
             new_val =p.number_to_words(val)
             return str(new_val)
    except ValueError:
        pass

    # Return the original if no transformation was applied
    return original_str


global model, tokenizer
model, tokenizer = None, None

global selection_model, selection_tokenizer
selection_model, selection_tokenizer = None, None

user_prompt_modify_sent="SENTENCE: {claim} \n\nMODIFIED SENTENCE:"
user_prompt_gen_doc="CLAIM: {claim} \n\nGENERATED DOCUMENT:"
user_prompt_modify_sent_with_doc= "SENTENCE: {claim} \n\nDOCUMENT: {doc} \n\nMODIFIED SENTENCE:"
### user_prompt_modify_doc="DOCUMENT: {doc} \n\n MODIFIED DOCUMENT:"
user_prompt_modify_doc_with_sent="DOCUMENT: {doc} \n\nCLAIM: {claim} \n\nMODIFIED DOCUMENT:"
user_prompt_gen_doc_with_sent="DOCUMENT: {doc} \n\nCLAIM: {claim} \n\nGENERATED DOCUMENT:"

regex = r'\b(\d{1,4}(?:,\d{3})*(?:\.\d+)?|(?:\d{1,2}[-/])?\d{1,2}[-/]\d{2,4}|\d+\s*\%)\b'


# sentence -> sentence
# sentence -> new doc
# doc -> modified doc   modify_doc
# doc sentence -> new doc  gen_doc_with_sent
# doc sentence -> modified doc

# ="Given a claim, generate a 100-word document. The main content of the document should elaborate on the claims and contain the main content of the claim."
# ="Given a claim, generate a 100-word document. The main content of the document should be different from the claims and only briefly mentions the claim"


#### New prompts:
# Generate a claim that includes information that cannot be verified within the provided context.
#Generate a claim that contradicts the provided context.

gen_prompt= {
    "from_claim":{
        "negation":{
            "system":"Give a negative or a contradicting form of this sentence while maintaining the same formulation of the sentence. If not possible answer with 'Pass'. You should directly provide the modified sentence or 'Pass' without explanation or added comment.",
            "user":user_prompt_modify_sent,
            "attribution_label": "change",
            },
        "reformulation":{  
            "system":"Give a full reformulation of this sentence while maintaining the exact same facts mentioned. If not possible answer with 'Pass'. You should directly provide the modified sentence or 'Pass' without explanation or added comment.",
            "user":user_prompt_modify_sent,
            "attribution_label": "keep",
            },

        "erronous_change":{ 
            "system":"Make a slight change to this sentence to falsify it. Change one important word or two to change the fact. For example replace units by different unequivalent units. If not possible answer with 'Pass'. You should directly provide the modified sentence or 'Pass' without explanation or added comment.",
            "user":user_prompt_modify_sent,
            "attribution_label": "change",
        },

        "infer_claim":{ 
            "system":"Given a claim, generate a 150-word document from which the claim can be extrapolated or inferred. The document should NOT directly support the claim but should contain information that can help infer it. If not possible answer with 'Pass'. You should directly provide the generated content or 'Pass' without explanation or added comment.",
            "user":user_prompt_gen_doc,
            "attribution_label": "keep", 
        },

        "over_infer_claim":{ 
            "system":"Given a claim, generate a 150-word document that is a very vague and general extrapolation of the claim. The document should NOT directly support the claim, mention it or refute it but contain vaguely similar to the given claim. The generated document should be a very vague generalization on the given claim. If not possible answer with 'Pass'. You should directly provide the generated content or 'Pass' without explanation or added comment.",
            "user":user_prompt_gen_doc,
            "attribution_label": "change", 
        },

        "combine_facts":{ 
            "system":"Given the following sentence, first determine atomic claims made in this sentence. Then for each claim generate a 150-word document that support it. The main content of each document should only tackle one claim and the main topic of the document does not have to be the claim itself. If not possible answer with 'Pass'. You should directly provide the generated documents as a list or 'Pass' without explanation or added comment.",
            "user":user_prompt_gen_doc,
            "attribution_label": "keep",
        },

        "add_to_the_claim_contradicting_info":{ 
            "system":"Given a claim, and a supporting document, add to the claim a new information. The added information should be contradicting something in the document. Make sure to keep the claim exactly as it is plus the added information. The added information should be short. If not possible or if the claim or document are too complex answer with 'Pass'. You should directly provide the modified claim or 'Pass' without explanation or added comment.",
            "user":user_prompt_modify_sent_with_doc,
            "attribution_label": "change",
        },
        "general_infer":{ 
            "system":"Given a sentence, generate a new sentence that is vaguely similar to the given sentence. The generated sentence should be a very vague generalization on the given sentence. If not possible or if the claim or document are too complex answer with 'Pass'. You should directly provide the modified claim or 'Pass' without explanation or added comment.",
            "user":user_prompt_modify_sent,
            "attribution_label": "change",
        },#
    },
    "from_passage":{ 
        "add_relevant_to_claim":{ 
            "system":"Given a claim, generate a 150-word a relevant document. The document must be relevant to the very general topic but completely without mention of the claim. Do not mention, support, or refute the claim—just describe the context around it. If not possible or if the claim is too complex answer with 'Pass'. You should directly provide the document or 'Pass' without explanation or added comment.",
            "user":user_prompt_gen_doc, 
            "attribution_label": "keep",
        },
        "add_irrelevant_to_claim":{ 
            "system":"Given a claim, generate a 150-word document containing irrelevant and neutral information. The added information should be genral and neutral to the claim (meaning does NOT mention, support, refute, or contradict it). If not possible or if the claim is too complex answer with 'Pass'. You should directly provide the document or 'Pass' without explanation or added comment.",
            "user":user_prompt_gen_doc, 
            "attribution_label": "keep",
        }, 
        "add_contradiction":{
            "system":"Given a claim, and a supporting document, generate a 100-word new document that contains contradicting information. The information in the document should contradict the claim. If not possible or if the claim or document are too complex answer with 'Pass'. You should directly provide the generated document or 'Pass' without explanation or added comment.",
            "user":user_prompt_gen_doc_with_sent,
            "attribution_label": "change",
        },
        "add_conflicting_sources":{ 
            "system":"Given a claim, and a supporting document, generate a 200-word new document that contains conflicting information regarding the claim. If not possible or if the claim or document are too complex answer with 'Pass'. You should directly provide the generated content or 'Pass' without explanation or added comment.",
            "user":user_prompt_gen_doc_with_sent,
            "attribution_label": "change",
        },
    },
}

fewshot_examples = {
    "from_claim":{
        "negation":[
            {
                "query":"",
                "sentence":"These barriers, in turn, can affect the volume and distribution of international trade, as well as the composition of exports and imports among countries",
                "modified":"These barriers are inaffectual to the volume and distribution of international trade, nor the composition of exports and imports among countries",
                "attribution_label": "change",
                "prompt_type":"modify_sent",
                
            },
            {
                "query":"",
                "sentence":"Lake Victoria has a maximum depth of 80 meters",
                "modified":"Lake Victoria has a minimum depth of 80 meters",
                "attribution_label": "change",
                "prompt_type":"modify_sent",
                
            },
            {
                "query":"",
                "sentence":"Pearl beer is no longer in production",
                "modified":"Pearl beer is still running in production",
                "attribution_label": "change",
                "prompt_type":"modify_sent",
                
            }

            

        ],
        "reformulation":[],
        "neutral_change":[
            {
                "query":" ",
                "sentence":"they also sell some 7,000 candies from around the world",
                "modified":"they also stock some 7,000 candies from around the world",
                "attribution_label": "keep",
                "prompt_type":"modify_sent",
                
            },
            {
                "query":" ",
                "sentence":"Walter Law was born in England.",
                "modified":"Walter William Law was born in England..",
                "attribution_label": "keep",
                "prompt_type":"modify_sent",
                
            },

        ],
        "erronous_change":[
            {
                "query":" the boiling point of water at 7000 meters above sea level",
                "sentence":"at 7,000 feet water boils 92.7C (198.9F).",
                "modified":"at 7,000 meters water boils 92.7C (198.9F).",
                "attribution_label": "change",
                "prompt_type":"modify_sent",
                
            },
            {
                "query":" Song Again by YUI",
                "sentence":"The song “Again” by YUI was used as the first opening theme song for the Fullmetal Alchemist: Brotherhood anime series",
                "modified":"The song “Again” by Dorcas Cochran was used as the first opening theme song for the Fullmetal Alchemist: Brotherhood anime series",
                "attribution_label": "change",
                "prompt_type":"modify_sent",
            },
            {
                "query":"",
                "sentence":"they also sell some 7,000 candies from around the world",
                "modified":"they also buy some 7,000 candies from around the world",
                "attribution_label": "change",
                "prompt_type":"modify_sent",
            },
            {
                "query":"",
                "sentence":"Dawn and Tony Orlando sing 'Tie a Yellow Ribbon Around the Old Oak Tree.",
                "modified":"Dawn and Tony Orlando wrote 'Tie a Yellow Ribbon Around the Old Oak Tree.",
                "attribution_label": "change",
                "prompt_type":"modify_sent",
            },
            {
                "query":" Why people living in equator have dark skin than people far from equator.",
                "sentence":"dark-skinned humans now living well north or south of the equator were receiving far less UV light than their equator-living ancestors.",
                "modified":"dark-skinned humans now living well north or south of the equator were receiving far less UV light than their equator-living counterparts.",
                "attribution_label": "change",
                "prompt_type":"modify_sent",
            }
            
        ],
        "doubtful":[
            {
                "query":"who was the book of first john written to",
                "sentence":" The Book of 1 John was likely written between A.D. 85-95.",
                "modified":"The book of 1 John was written around A.D. 85-95.",
                "attribution_label": "reduce_score", 
                "prompt_type":"modify_sent",
            },
            {
                "query":"",
                "sentence": "Some people believe that regulation is necessary to ensure equal treatment of all internet communications.",
                "modified":"Regulation is definitely necessary to ensure equal treatment of all internet communications.",
                "attribution_label": "reduce_score",
                "prompt_type":"modify_sent", 
            },
            {
                "query":" when does the fiscal year begin and end",
                "sentence": "the first quarter of the fiscal year 2023 is from October 1, 2022 to December 31, 2022.",
                "modified":"Pass",
                "attribution_label": "reduce_score",
                "prompt_type":"modify_sent", 
            },
        ],
        "infer_claim":[],

        "over_infer_claim":[
            {
                "query":"",
                "sentence":"Drinking peppermint tea after dinner improves the quality of deep sleep in adults",
                "document":"Various botanical infusions have historically been associated with the promotion of physiological balance during periods of rest. As individuals explore different dietary habits to enhance their nightly routines, the integration of natural elements into the evening cycle often reflects a desire for improved physical tranquility. Generally, the consumption of organic derivatives can influence the internal environment, potentially facilitating a more stable state of relaxation as the body transitions into its natural restorative phases.",
                "prompt_type":"gen_doc_from_sent",
            },
            {
                "query":"",
                "sentence":"NASA cancelled the Apollo program because of the high financial costs of moon exploration.",
                "document":"The cessation of a high-profile aerospace initiative is frequently dictated by the shifting allocation of national fiscal resources and the evolving priorities of governmental oversight. As the economic burden of maintaining complex exploratory ventures increases, administrative bodies must evaluate the long-term viability of continued investment against other competing budgetary demands. The withdrawal of funding for specialized scientific endeavors often reflects a strategic pivot toward more cost-effective or terrestrial objectives. In general, the termination of expansive technological projects is a direct result of the tension between ambitious scientific goals and the pragmatic constraints of a nation's monetary framework.",
                "prompt_type":"gen_doc_from_sent",
            }
        ],
        "gen_supporting_doc":[],
        "general_infer":[],
        "combine_facts":[
            {
                "query":"",
                "sentence":"The first overseas branch of Bible Students was opened in London in 1900, and a German branch office of the Watch Tower Society opened in Elberfeld in 1902.",
                "document":["Jehovah's Witnesses were an outgrowth of the International Bible Students. A German branch office of the Watch Tower Society opened in Elberfeld in 1902. By 1933, almost 20,000 Witnesses were counted as active door-to-door preachers, and their annual Memorial service was attracting almost 25,000 people. In Dresden, there were more Bible Students than in New York, where the Watch Tower Society was headquartered.", "Jehovah's Witnesses originated as a branch of the Bible Student movement, which developed in the United States in the 1870s among followers of Christian Restorationist minister Charles Taze Russell. Bible Student missionaries were sent to England in 1881 and the first overseas branch was opened in London in 1900. The group took on the name International Bible Students Association and by 1914 it was also active in Canada, Germany, Australia and other countries. The movement split into several rival organizations after Russell's death."],
                "attribution_label": "keep",
                "prompt_type":"gen_doc_from_sent",
            },
            {
                "query":"",
                "sentence":"The James Webb Space Telescope, a ten billion dollar orbiting observatory, is capturing and transmitting unprecedented images of the deep universe.",
                "document":["The James Webb Space Telescope (JWST) represents the pinnacle of human engineering and international collaboration. As a ten billion dollar billion endeavor led by NASA, with contributions from the ESA and CSA, it is the most complex and powerful space observatory ever built. Launched in late 2021, the JWST was designed to succeed the Hubble Space Telescope, offering significantly increased sensitivity and resolution.", "The operational phase of the James Webb Space Telescope has ushered in a new era of astronomy, characterized by the capture and transmission of data that was previously invisible to human eyes. By utilizing advanced mid-infrared and near-infrared instruments, the telescope can peer through dense clouds of cosmic dust that obscure star-forming regions. This capability allows it to transmit high-definition imagery of 'stellar nurseries' and the intricate structures of distant nebulae with startling clarity."],
                "attribution_label": "keep",
                "prompt_type":"gen_doc_from_sent",
            },
            {
                "query":"",
                "sentence":"The Colosseum in Rome, a nearly two thousand year-old stone amphitheater, is crumbling and undergoing extensive restoration.",
                "document":["The Flavian Amphitheatre, universally known as the Colosseum, stands as the most iconic symbol of Imperial Rome. Completed in 80 AD under the Emperor Titus, this nearly two thousand-year-old structure remains the largest standing amphitheater in the world. Built primarily of travertine limestone, tuff (volcanic rock), and brick-faced concrete, it represents a monumental achievement in Roman engineering. The elliptical design was a departure from the traditional semi-circular Greek theaters, allowing for a massive seating capacity of up to eighty thousand spectators.","Despite its enduring presence, the Colosseum faces a modern battle against time, environmental decay, and structural instability. The crumbling of this ancient monument is a result of nearly two millennia of stressors, ranging from historical lightning strikes and earthquakes to contemporary traffic vibrations and acid rain. The porous travertine stone is particularly susceptible to 'black crusting,' a chemical reaction caused by urban pollution that eats away at the surface and weakens the structural integrity of the outer arches."],
                "attribution_label": "keep",
                "prompt_type":"gen_doc_from_sent",
            },

        ],
        "supporting_irrelevant_content":[
            {
                "query":"",
                "sentence":"ESPN won an emmy for the creation of the yellow line.",
                "document":"Football is a family of team sports that involve, to varying degrees, kicking a ball to score a goal. Unqualified, the word football is understood to refer to whichever form of football is the most popular in the regional context in which the word appears. Sports commonly called football in certain places include association football ( known as soccer in some countries ) ; gridiron football ( specifically American football or Canadian football ) ; Australian rules football ; rugby football ( either rugby league or rugby union ) ; and Gaelic football. These different variations of football are known as football codes. the University of Iowa's locker room for visiting football teams is completely painted pink. in the 1960's top bowlers made twice as much as top football stars. - That the highest score ever in a football game occurred in 1916 when Georgia Tech defeated Cumberland 222 - 0. ESPN won an emmy for the creation of the superimposed yellow line representing the first down line for American football games. Former Partiots RB BenJarvus Green - Ellis has never fumbled the football in his NFL career.",
                "attribution_label": "keep",
                "prompt_type":"gen_doc_from_sent",
            },
            {
                "query":"",
                "sentence": "Seismologists can also directly compare the shape of the waves recorded from each test to determine whether they are similar.",
                "document": "What earthquake science can tell us about North Korea's nuclear test. There are a number of ways to do this. One is to measure the depth at which the earthquake occurred. Even with modern drilling technology, it is only possible to place a nuclear device a few kilometres below the ground; if an earthquake occurs at a depth of more than 10km, we can be certain it is not a nuclear explosion. Studies of the numerous nuclear tests that took place during the Cold War show that explosions generate larger P waves than S waves when compared with earthquakes. Explosions also generate proportionally smaller Surface waves than P waves. Seismologists can therefore compare the size of the different types of wave to try to determine whether the waves came from an explosion or a natural earthquake. For cases like North Korea, which has carried out a sequence of nuclear tests since 2006, we can directly compare the shape of the waves recorded from each test. As the tests were all conducted at sites within a few kilometres of each other, the waves have a similar shape, differing only in magnitude.",
                "attribution_label":"keep",
                "prompt_type":"gen_doc_from_sent",
            },

        ],
        "unsupporting_relevant_content":[],

        "add_to_the_claim_neutral_info":[
            {
                "query":"",
                "sentence": "Elon Musk completed his $44 billion deal to buy Twitter",
                "document":"The world’s richest man, Elon Musk, has completed his $44bn (£38.1bn) takeover of Twitter, according to a filing with the US government.",
                "modified":"Elon Musk completed his $44 billion deal to buy Twitter on October 27, 2022",
                "attribution_label": "reduce_score",
                "prompt_type":"modify_sent_with_doc",
            },
            {
                "query":"",
                "sentence": "the U.S. trade deficit increased from $676.7 billion in 2020 to $859.1 billion in 2021",
                "document":"The U.S. trade deficit increased from $676.7 billion in 2020 to $859.1 billion in 2021.",
                "modified":"According to The Balance, the U.S. trade deficit increased from $676.7 billion in 2020 to $859.1 billion in 2021",
                "attribution_label": "reduce_score",
                "prompt_type":"modify_sent_with_doc",
            },
        ],
        "add_to_the_claim_contradicting_info":[],
    },
    "from_passage":{
        "add_relevant_to_claim":[
            {
                "query":"",
                "sentence":"The use of bio-luminescent fungi in urban lighting reduces the total electricity consumption of municipal street grids.", #modified_document
                "document":"The study of bio-luminescent organisms involves a deep analysis of the chemical reactions between luciferin and the enzyme luciferase. Within various fungi species, this reaction results in a consistent green glow that emanates from the gills and mycelium. When these fungi are introduced into urban environments, such as public parks or botanical gardens, the primary concern is maintaining a steady supply of decaying organic matter to sustain their growth.",
                "prompt_type":"gen_doc_from_sent",
            },

        ],
        "add_irrelevant_to_claim":[ 
            {
                "query":"",
                "sentence":"ESPN won an emmy for the creation of yellow line", 
                "document":"Football is a family of team sports that involve, to varying degrees, kicking a ball to score a goal. Unqualified, the word football is understood to refer to whichever form of football is the most popular in the regional context in which the word appears. Sports commonly called football in certain places include association football ( known as soccer in some countries ) ; gridiron football ( specifically American football or Canadian football ) ; Australian rules football ; rugby football ( either rugby league or rugby union ) ; and Gaelic football. These different variations of football are known as football codes. the University of Iowa's locker room for visiting football teams is completely painted pink. in the 1960's top bowlers made twice as much as top football stars. - That the highest score ever in a football game occurred in 1916 when Georgia Tech defeated Cumberland 222 - 0.", # ESPN won an emmy for the creation of the superimposed yellow line representing the first down line for American football games. Former Partiots RB BenJarvus Green - Ellis has never fumbled the football in his NFL career.",
                "attribution_label": "keep",
                "prompt_type":"gen_doc_from_sent",
            },
            {
                "query":"Why does the President of the United States need to be born in the United States to be eligible to run?",
                "sentence": "This means that the President must be born in the United States, or its territories, or on a military base, to parents who were both citizens of the United States.",
                "document":"The framers were genuinely afraid of foreign subversion. Among their nightmare scenarios was the prospect of a European noble using his money and influence to sway the Electoral College, take command of the American army, and return the nascent nation to the royalist fold. At the time, several European figures—such as France’s Marquis de Lafayette, a hero of the Revolutionary War—were quite popular in the New World, so the idea wasn’t completely far-fetched.", #The presidential birth requirements in the U.S. Constitution require anyone elected to serve as U.S. president or vice president be a 'natural born citizen.' What that means is only those people who are U.S. citizens at birth and did not have to go through the naturalization process are eligible to serve in the highest office in the land. It does not mean that a president must have been born on U.S. soil to serve, even though there has never been a U.S. president born outside one of the 50 U.S. states.
                "attribution_label": "keep",
                "prompt_type":"gen_doc_from_sent", 
                
            },
        ],
        "add_conflicting_sources":[
            {
                "query":"What’s the population of Tanzania?",
                "document":"Tanzania has a current population of 55.57 million people. Current statistics form the World Bank show that in 2011, 49.1 of Tanzanians lived below $1.90 USD per day. This figure is an improvement over 2007’s report indicating a poverty rate of 55.1%. Tanzania has seen annual GDP gains of 7 since 2010 and this economic growth is attributed to this positive trends for poverty alleviation in Tanzania.'",
                "sentence":"The population of Tanzania is 55.57 million people.",
                "generated_document":"Tanzania is the most populous and vast country in East Africa with a population of 50.1 million people. It is a sparsely populated country with a geographically wide distribution of settlements hence presenting a challenge regarding access to hospitals. With its population and an area of 940,000km, its population density varies from 12 people per km in Lindi to 3,133 people per km in Dar es salaam.. There are 26 administrative regions in the country and the list of hospitals will be arranged by regions",
                "attribution_label": "change",
                "prompt_type":"gen_doc_with_sent",
                    
            },
            {
                "query":"",
                "document":"Taylor Swift’s 'Reputation' was the best selling album of 2017. Singer Ed Sheeran\’s '÷' is the second best-selling album with 451,000 album-equivalent units, of which 322,000 were pure album sales. Rapper Kendrick Lamar’s fourth studio album, 'Damn', went on to become the third best-seller with 603,000 album-equivalent units in its first week of release (353,000 in pure album sales)",
                "sentence":"The best selling album of 2017 was Taylor Swift’s 'Reputation'",
                "generated_document":"Reputation received generally positive reviews from music critics and reached number one in 13 countries including the United Kingdom, and United States. in the US, the album sold 1.216\xa0million copies in its first week of release, making it the country’s best-selling album of 2017, while with global sales of 4.5\xa0million copies, it was the second best-selling album of 2017 worldwide.",
                "attribution_label": "change",
                "prompt_type":"gen_doc_with_sent",
                    
            },
        ],
        "add_contradiction":[],
    }
}

#all types: reformulation negation erronous_change infer_claim combine_facts supporting_irrelevant_content unsupporting_relevant_content add_to_the_claim_neutral_info "add_to_the_claim_contradicting_info" "add_relevant_to_claim" "add_irrelevant_to_claim" "add_contradiction" "add_conflicting_sources"

def prepare_prompt(claim=None, passage=None, prompt=None, prompt_type=None, fewshot_examples=[], model_name = "meta-llama/Llama-2-13b-chat-hf", nb_examples=1):    
    global model, tokenizer
    if model is None:
         #"stabilityai/stablelm-zephyr-3b" #"meta-llama/Llama-2-7b-chat-hf"
        logger.info(f"Loading Language model...{model_name}")
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name) #, cache_dir= os.environ['WORK'] + '/.cache/huggingface/hub')

        model = LLM(
            model=model_name,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.9,
            max_model_len=MAX_MODEL_LEN,
            dtype='bfloat16',
            enforce_eager=False,
            # for LORA
            trust_remote_code=True,
            enable_lora=True,
        )

    user_prompt = prompt["user"]
    system_prompt=prompt["system"]
    if claim:
        user_prompt = prompt["user"].replace("{claim}", claim)
    if passage:
        user_prompt = user_prompt.replace("{doc}", passage)

    generation_start_token="MODIFIED SENTENCE"
    inputs=[]
    if fewshot_examples:
        formatted_examples = []
        selected_examples = fewshot_examples
            # selected_examples.reverse()
        for example in selected_examples:
                # sentence -> sentence
                # sentence -> new doc  gen_doc_from_sent
                # doc -> modified doc   modify_doc
                # doc sentence -> new doc  gen_doc_with_sent
                # doc sentence -> modified doc
                #             user_prompt_modify_sent="SENTENCE: {claim} \n\n MODIFIED SENTENCE:"    modify_sent
                # user_prompt_gen_doc="CLAIM: {doc} \n\n GENERATED DOCUMENT:"   gen_doc_from_sent
                # user_prompt_modify_sent_with_doc= "SENTENCE: {claim} \n\n DOCUMENT: {doc} \n\n MODIFIED SENTENCE:"  modify_sent_with_doc
                # ### user_prompt_modify_doc="DOCUMENT: {doc} \n\n MODIFIED DOCUMENT:"   modify_doc
                # user_prompt_modify_doc_with_sent="DOCUMENT: {doc} \n\n CLAIM: {doc} n\n MODIFIED DOCUMENT:"   gen_doc_with_sent
            if  example["prompt_type"] == "modify_sent": #query and "query" in example.keys() and
                if  "modified" in example.keys():
                    formatted_example = (
                        "SENTENCE: " + example["sentence"] + "\n\nMODIFIED SENTENCE: " + example["modified"]
                    )
                    generation_start_token="MODIFIED SENTENCE:"
            elif example["prompt_type"] == "gen_doc_from_sent":
                add_s= "S" if len(example["document"]) >= 1 else ""  
                formatted_example = (
                    "CLAIM: " + example["sentence"] + "\n\nGENERATED DOCUMENT"+add_s+": " + str(example["document"])
                )
                generation_start_token="GENERATED DOCUMENT:"
            elif example["prompt_type"] == "modify_sent_with_doc":
                formatted_example = (
                    "SENTENCE: " + example["sentence"] + "\n\nDOCUMENT:" + example["document"] + "\n\nMODIFIED SENTENCE"+ example["modified"]
                )
                generation_start_token="GENERATED DOCUMENT:"
            elif example["prompt_type"] == "modify_doc":
                if  "document" in example.keys():
                    formatted_example = (
                        "DOCUMENT: " + example["document"] + " \n\nMODIFIED DOCUMENT: \n"+ example["modified_document"]
                    )
                    generation_start_token="MODIFIED DOCUMENT:"
            elif example["prompt_type"] == "modify_doc_with_sent":
                if  "document" in example.keys():
                    formatted_example = (
                        "DOCUMENT: " + example["document"] + "\n\nCLAIM: " + example["sentence"] +" \n\nMODIFIED DOCUMENT: \n"+ example["modified_document"]
                    )
                    generation_start_token="MODIFIED DOCUMENT:"
                
            elif example["prompt_type"] == "gen_doc_with_sent":
                if  "document" in example.keys():
                    formatted_example = (
                        "DOCUMENT: " + example["document"] + "CLAIM: " + example["sentence"] + " \n\nGENERATED DOCUMENT: \n"+ example["generated_document"]
                    )
                    generation_start_token="GENERATED DOCUMENT:"
            else:
                logger.warning(f"Fewshot example does not contain a sentence nor document: {example}")
                formatted_example=""
            formatted_examples.append(formatted_example)

        complete_user_prompt = "\n\n".join(formatted_examples) + "\n\n" + user_prompt
        user_prompt = complete_user_prompt

    input_text = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", 
            "content": user_prompt}
        ]
    inputs = tokenizer.apply_chat_template(
            input_text, add_generation_prompt=True, return_tensors="pt",tokenize=False
        )
    return inputs, generation_start_token



def generate_vllm(all_inputs,all_start_tokens,qwen=False):
    global model, tokenizer
    if model is None:
         #"stabilityai/stablelm-zephyr-3b" #"meta-llama/Llama-2-7b-chat-hf"
        logger.info(f"Loading Language model...{model_name}")
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name) #, cache_dir= os.environ['WORK'] + '/.cache/huggingface/hub')

        model = LLM(
            model=model_name,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.9,
            max_model_len=MAX_MODEL_LEN,
            dtype='bfloat16',
            enforce_eager=False,
            # for LORA
            trust_remote_code=True,
            enable_lora=True,
        )


    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=MAX_NEW_TOKENS,
        truncate_prompt_tokens=MAX_PROMPT_FOR_TRUNCATION,
        #stop_token_ids=[151643, 151645]
    )
    all_outputs=model.generate(all_inputs, sampling_params) # [inputs]

    all_answers=[]
    for idx_output, output in enumerate(all_outputs):
        #original_prompt = output.prompt
    
        answer = output.outputs[0].text

        generation_start_token = all_start_tokens[idx_output]
        keyword = ":assistant\n" #"[/INST]"
        index_kw = answer.rfind(keyword)
        filetred_answer = answer
        if index_kw != -1:
            filetred_answer = answer[index_kw+len(keyword)+1:]
        else:
            generated_content_index = filetred_answer.rfind(generation_start_token)
            if generated_content_index != -1:
                filetred_answer = filetred_answer[generated_content_index+len(generation_start_token)+1:]
            else:
                generated_content_index = filetred_answer.rfind("\n\n")
                if generated_content_index !=-1:
                    filetred_answer = filetred_answer[generated_content_index+len("\n\n")+1:]

        logger.info(f"Generated content: {filetred_answer}")
        all_answers.append(filetred_answer)
    return all_answers


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name for generating queries"
    ) #meta-llama/Llama-3.1-8B-Instruct,  meta-llama/Llama-2-13b-chat-hf
    parser.add_argument(
        "--data_file", type=str, default=None, help="Dataset (.json)", required=True,
    )

    parser.add_argument(
        "--prompt_type", type=str, default="all", help="prompt used", choices=["all","over_infer_claim","numerical_mismatch","reformulation", "negation", "erronous_change", "infer_claim", "combine_facts", "supporting_irrelevant_content", "unsupporting_relevant_content", "add_to_the_claim_neutral_info", "add_to_the_claim_contradicting_info", "add_relevant_to_claim", "add_irrelevant_to_claim", "add_contradiction", "add_conflicting_sources"]
    ) 
    parser.add_argument(
        "--data_source", type=str, default=None, help="Specific dataset source"
    )
    parser.add_argument(
        "--results_folder", type=str, default="results/augmented_datasets/", help="Results folder"
    )
    parser.add_argument(
        "--validating_code", action="store_true",  help="validate_code"
    )
    parser.add_argument(
        "--zeroshot", action="store_true",  help="add or not fewshot examples"
    )
    parser.add_argument(
        "--resume_from_file", type=str, default=None, help="Resume from file"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batching dataset"
    )
    parser.add_argument(
        "--tag", type=str, default=None, help="add tag"
    )
    parser.add_argument(
        "--batched", action="store_true",  help="in batches"
    )
    parser.add_argument(
        "--startindex",
        type=int,
        default=0,
        help="Index to start iterations",
    )
    parser.add_argument(
        "--stopindex",
        type=int,
        default=None,
        help="Index to stop iterations",
    )

    args = parser.parse_args()

    #################
    # Logging params
    #################
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)} - {parser.get_default(arg)}")

    with open(args.data_file, 'r') as file:
        dataset = json.load(file)
        if isinstance(dataset, dict):
            dataset = dataset["data"]
    updated_dataset = []

    prompt_model_name = args.prompt_model_name.split('/')[-1]
    code_validation="_validating_code" if args.validating_code else ''
    prompt_type="_"+args.prompt_type if args.prompt_type else ''
    shots= "_zeroshot" if args.zeroshot else 'fewshot'
    tag = "_"+args.tag if args.tag else ""
    batched= "_batched" if args.batched else ""
    results_file =  "augmented_dataset"+prompt_model_name+prompt_type+shots+code_validation+tag+".json"
    results_folder=os.path.join(args.results_folder , prompt_model_name+prompt_type+shots+code_validation+tag)
    inter_results_folder=os.path.join(results_folder, "intermediate")

    try:
        logger.info(f"MAKING NEW FOLDER: {args.results_folder}")
        os.makedirs(results_folder, exist_ok=True)
        logger.info(f"Directory '{results_folder}' created successfully or already exists.")

        logger.info(f"MAKING Intermediate FOLDER: {inter_results_folder}")
        os.makedirs(inter_results_folder, exist_ok=True)
        logger.info(f"Directory '{inter_results_folder}' created successfully or already exists.")
    except Exception as e:
        logger.info(f"An error occurred: {e}")

    start_idx = args.startindex 
    if args.resume_from_file:
        with open(args.resume_from_file) as f:
            data_with_config = json.load(f)
        updated_dataset = data_with_config["data"]
        start_idx = start_idx + len(updated_dataset)
        logger.info(f"Resuming test from file: {args.resume_from_file}")
    logger.info(f"Starting Iteration: {start_idx}")

    for idx, row in enumerate(tqdm(dataset)):
        if idx < start_idx:
            continue
        if args.validating_code and  idx == 2:
             break
        negative_examples=[]
                            
        logger.info(f"Generating Negative Examples")
        if row["attribution_label"] == "attributable":
             #################### ATTRIBUTABLE both models correct
            if row["nli_score"]==1 and row["align_score"]>=0.6:
                row["example_type"]="hard_positive"

                ############# NUMERICAL VALUES
                if args.prompt_type =="all" or args.prompt_type == "numerical_mismatch":
                    if re.search(regex,row["claim"]):
                        added_example=copy.deepcopy(row)
                        logger.info(f"Claim contains numerical value: {row['claim']}")
                        generated_content = re.sub(regex, create_wrong_value, row["claim"])
                        logger.info(f"Numerical Mismatch: {generated_content}")
                        added_example["example_type"]="generated"
                        added_example["claim"]=generated_content                            
                        added_example["original_label"]=row["attribution_label"]
                        added_example["label_switch"]="change"
                        added_example["attribution_label"] = "not attributable"
                        added_example["error_type"]="claim_numerical_mismatch"
                        if 'align_score' in added_example.keys() and 'nli_score' in added_example.keys():
                            del added_example["nli_score"]
                            del added_example["align_score"]
                        negative_examples.append(added_example)



                        added_example=copy.deepcopy(row)
                        generated_content = re.sub(regex, transcript_numbers, row["claim"])
                        logger.info(f"Numerical Trascript into letters: {generated_content}")
                        added_example["example_type"]="generated"
                        added_example["claim"]=generated_content                            
                        added_example["original_label"]=row["attribution_label"]
                        added_example["label_switch"]="keep"
                        added_example["attribution_label"] = "attributable"
                        added_example["error_type"]="claim_numerical_transcript"
                        if 'align_score' in added_example.keys() and 'nli_score' in added_example.keys():
                            del added_example["nli_score"]
                            del added_example["align_score"]
                        negative_examples.append(added_example)


                ########################### CLAIM TRANSFORMATION
                all_inputs=[]
                all_generated_outputs=[]
                all_start_tokens=[]
                for error_type in gen_prompt["from_claim"].keys():
                    if args.prompt_type =="all" or args.prompt_type == error_type:
                        if error_type in ["add_to_the_claim_neutral_info","add_to_the_claim_contradicting_info"]:
                            passges= " ".join(row["references"])
                        else:
                            passges= None

                        logger.info(f"Logging Example Generation: from_claim - {error_type}")
                        fewshots=fewshot_examples["from_claim"][error_type]
                        if args.zeroshot:
                            fewshots =[]
                        inputs, generation_start_token= prepare_prompt(row["claim"], passage=passges, prompt=gen_prompt["from_claim"][error_type], fewshot_examples=fewshots, model_name =args.prompt_model_name)
                        all_inputs.append(inputs)
                        all_start_tokens.append(generation_start_token)
                if len(all_inputs):
                    all_generated_outputs = generate_vllm(all_inputs,all_start_tokens)
                #all_generated_outputs = generate_example(row["claim"], passage=row["references"], prompt=gen_prompt["from_claim"], fewshot_examples=fewshot_examples["from_claim"],  model_name =args.prompt_model_name)
                for idx_content, generated_content in enumerate(all_generated_outputs):
                    error_type=list(gen_prompt["from_claim"].keys())[idx_content]
                    logger.info(f"claim: {row['claim']}")
                    logger.info(f"Passages: {passges}")
                    logger.info(f"Generated_content: {generated_content}")
                    if generated_content != "Pass":
                        added_example= copy.deepcopy(row)
                        if error_type in ["infer_claim",  "combine_facts", "over_infer_claim","supporting_irrelevant_content", "unsupporting_relevant_content"]: 
                            added_example["references"]=[generated_content]
                        else:
                            added_example["claim"]=generated_content
                        added_example["original_label"]=row["attribution_label"]
                        added_example["label_switch"]=gen_prompt["from_claim"][error_type]["attribution_label"]
                        added_example["attribution_label"] = "attributable" if gen_prompt["from_claim"][error_type]["attribution_label"] == "keep" else "not attributable"
                        added_example["error_type"]="claim_"+error_type
                        added_example["example_type"]="generated"
                        if 'align_score' in added_example.keys() and 'nli_score' in added_example.keys():
                            del added_example["nli_score"]
                            del added_example["align_score"]
                        negative_examples.append(added_example)


                all_inputs=[]
                all_generated_outputs=[]
                all_start_tokens=[]
                all_error_types=[]
                modified_docs=[]
                for error_type in gen_prompt["from_passage"].keys():
                    if args.prompt_type =="all" or args.prompt_type == error_type:
                        logger.info(f"Logging Example Generation: from_passage - {error_type}")
                        fewshots=fewshot_examples["from_passage"][error_type]
                        if args.zeroshot:
                            fewshots =[]
                        inputs, generation_start_token= prepare_prompt(row["claim"], passage="\n".join(row["references"]), prompt=gen_prompt["from_passage"][error_type], fewshot_examples= fewshots, model_name =args.prompt_model_name)
                        all_inputs.append(inputs)
                        all_start_tokens.append(generation_start_token)
                        all_error_types.append(error_type)
                if len(all_inputs):
                    all_generated_outputs = generate_vllm(all_inputs,all_start_tokens)
                for idx_content, generated_content in enumerate(all_generated_outputs):
                    error_type=all_error_types[idx_content]
                    logger.info(f"claim: {row['claim']}")
                    logger.info(f"Passages: {'\n'.join(row['references'])}")
                    logger.info(f"Generated_content: {generated_content}")
                    if generated_content != "Pass":
                        added_example=copy.deepcopy(row)
                        added_example["references"].append(generated_content)
                        random.shuffle(added_example["references"])
                        added_example["original_label"]=row["attribution_label"]
                        added_example["label_switch"]=gen_prompt["from_passage"][error_type]["attribution_label"]
                        added_example["attribution_label"] = "attributable" if  gen_prompt["from_passage"][error_type]["attribution_label"] == "keep" else "not attributable"
                        added_example["error_type"]="modify_passage-"+error_type
                        added_example["example_type"]="generated"
                        if 'align_score' in added_example.keys() and 'nli_score' in added_example.keys():
                            del added_example["nli_score"]
                            del added_example["align_score"]
                        negative_examples.append(added_example)

            else:
                row["example_type"]="unused"

        elif row["attribution_label"] == "not attributable":
            if row["nli_score"]==0 and row["align_score"]<0.6: ### both models correct
                row["example_type"]="hard_positive"

                all_inputs=[]
                all_start_tokens=[]
                all_generated_outputs=[]
                
                for error_type in ["reformulation"]: 
                    if args.prompt_type =="all" or args.prompt_type == error_type:
                        logger.info(f"Logging Example Generation: from_claim - {error_type}")
                        fewshots=fewshot_examples["from_claim"][error_type]
                        if args.zeroshot:
                            fewshots =[]
                        inputs, generation_start_token= prepare_prompt(row["claim"], passage=None, prompt=gen_prompt["from_claim"][error_type], fewshot_examples= fewshots, model_name =args.prompt_model_name)
                        all_inputs.append(inputs)
                        all_start_tokens.append(generation_start_token)
                if len(all_inputs):
                    all_generated_outputs = generate_vllm(all_inputs,all_start_tokens)
                for idx_content, generated_content in enumerate(all_generated_outputs):
                    error_type= "reformulation" #list(gen_prompt["from_claim"].keys())[idx_content]
                    logger.info(f"claim: {row['claim']}")
                    logger.info(f"Generated_content: {generated_content}")
                    if generated_content != "Pass":
                        added_example= copy.deepcopy(row)
                        if error_type in ["unsupporting_relevant_content"]: 
                            added_example["references"].append(generated_content)
                        else:
                            added_example["claim"]=generated_content
                        added_example["original_label"]=row["attribution_label"]
                        added_example["label_switch"]=gen_prompt["from_claim"][error_type]["attribution_label"]
                        added_example["attribution_label"] = "not attributable" if gen_prompt["from_claim"][error_type]["attribution_label"] == "keep" else "attributable"
                        added_example["error_type"]="modify_claim-"+error_type
                        added_example["example_type"]="generated"
                        negative_examples.append(added_example)



                    
                if args.prompt_type =="all" or args.prompt_type == "add_relevant_to_claim":
                    all_inputs=[]
                    all_start_tokens=[]
                    all_generated_outputs=[]
                    inputs, generation_start_token= prepare_prompt(row["claim"], passage="\n".join(row["references"]), prompt=gen_prompt["from_passage"]["add_relevant_to_claim"], fewshot_examples=fewshots, model_name =args.prompt_model_name)
                    all_inputs.append(inputs)
                    all_start_tokens.append(generation_start_token)
                    
                    if len(all_inputs):
                        all_generated_outputs = generate_vllm(all_inputs,all_start_tokens)
                    for idx_content, generated_content in enumerate(all_generated_outputs):
                        logger.info(f"claim: {row['claim']}")
                        logger.info(f"Passages: {'\n'.join(row['references'])}")
                        logger.info(f"Generated_content: {generated_content}")
                        if generated_content != "Pass":
                            added_example=copy.deepcopy(row)
                            added_example["references"].append(generated_content)
                            random.shuffle(added_example["references"])
                            added_example["original_label"]=row["attribution_label"]
                            added_example["label_switch"]=gen_prompt["from_passage"]["add_relevant_to_claim"]["attribution_label"]
                            added_example["attribution_label"] = "not attributable" if  gen_prompt["from_passage"]["add_relevant_to_claim"]["attribution_label"] == "keep" else "attributable"
                            added_example["error_type"]="modify_passage-add_relevant_to_claim"
                            added_example["example_type"]="generated"
                            negative_examples.append(added_example)


        updated_dataset.append(row)
        if len(negative_examples):                                     
            updated_dataset.extend(negative_examples)
        # if (idx+1) % 80 == 0:
        #     inter_results_file = "itermediate_" +  str(args.startindex)+results_file #"-"+str(idx)+ results_file
        #     inter_results_file = os.path.join(inter_results_folder, inter_results_file)
        #     new_set={"data":updated_dataset, "params":vars(args)}
        #     logger.info(f"Saving intermediate results to {inter_results_file}")
        #     with open(inter_results_file, "w") as writer:
        #         json.dump(new_set, writer)
        if args.stopindex and args.stopindex == idx:
            logger.info(f"Stoping after {args.stopindex - args.startindex + 1} iterartion, index {args.stopindex} finished")
            break
    if args.stopindex:
        #assert (args.stopindex == idx)
        inter_results_file = str(args.startindex)+"-"+str(args.stopindex)+"iter_"+results_file
        results_file = inter_results_file
    
    results_file = os.path.join(results_folder, results_file)
    new_set={"data":updated_dataset, "params":vars(args)}
    logger.info(f"Saving results to {results_file}")
    with open(results_file, 'w') as file:
        json.dump(new_set, file)

main()
