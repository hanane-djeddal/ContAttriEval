import re
from nltk import sent_tokenize
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging
import os
#from vllm import LLM, SamplingParams
#from alignscore import AlignScore


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# os.environ["HTTP_PROXY"] = "http://hacienda:3128"
# os.environ["HTTPS_PROXY"] = "http://hacienda:3128"

#os.environ['HF_HOME'] = os.environ['WORK'] + '/.cache/huggingface'
os.environ['HF_HOME'] = os.environ['WORK'] + '/.cache/huggingface'

AUTOAIS = "google/t5_xxl_true_nli_mixture" #"google/t5_xxl_true_nli_mixture"

global autoais_model, autoais_tokenizer
autoais_model, autoais_tokenizer = None, None

global alignscore
alignscore = None


def init_alignscore():
    global alignscore
    if alignscore is None:
        logger.info("Loading alignscore model...")
        alignscore = AlignScore(model='roberta-base', batch_size=32, device="cuda:0", ckpt_path='/home/djeddal/Documents/Code/RAGnRoll/model_ckpnt/AlignScore-large.ckpt', evaluation_mode='nli_sp')

def init_nli_dpo_aligned(model_path):
    global autoais_model, autoais_tokenizer
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16, #float32, #bfloat16, #
        dtype=torch.bfloat16, #float32, #bfloat16, 
    )
    logger.info(f'Loading model... {model_path}')
    autoais_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, 
        #torch_dtype=torch.bfloat16, 
        #torch_dtype=torch.float32,
        quantization_config=nf4_config,
    )
    autoais_tokenizer = AutoTokenizer.from_pretrained(
        model_path,  
        use_fast=False
    )

def init_tokenizers():
    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            dtype=torch.bfloat16,
            # bnb_4bit_compute_dtype=torch.float32,
            # dtype=torch.float32
        )
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(
            AUTOAIS,
            quantization_config=nf4_config,
        )
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS,use_fast=True)#, ) #

    ###### does not work with vllm    
    # if autoais_model is None:
    #     autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS, use_fast=False)
    #     autoais_model = LLM(
    #         model=AUTOAIS,
    #         tensor_parallel_size=torch.cuda.device_count(),
    #         gpu_memory_utilization=0.9,
    #         #max_model_len=model_max_len,
    #         dtype='bfloat16',
    #         enforce_eager=False,
    #         # for LORA
    #         trust_remote_code=True,
    #         enable_lora=True,
    #     )




def reposition_period_after_citation(text):
    result = re.sub(r'\.\s*((\[[^\]]+\])+)(?!\S)', r' \1.', text)
    return result

def remove_citations(sent,cited_docids=False):
    if cited_docids:
        citation_pattern = re.compile(r'\[\d+#\d+(?:\s*[-,]\s*\d+#\d+)*\](?:\[\d+#\d+\])*')
    else:
        citation_pattern =r'\[\d+(?:\s*[-,]\s*\d+)*\](?:\[\d+\])*'
    return re.sub(citation_pattern, "", sent) 


def get_statement_by_sentence(text):
    citation_pattern =r'\[\d+(?:\s*[-,]\s*\d+)*\](?:\[\d+\])*'
    number_pattern = re.compile(r"\d+")
    statements=[]

    sentences = sent_tokenize(text)
    for sentence in sentences:
        sentence = sentence.strip()
        src = re.findall(citation_pattern, sentence)
        int_src = list(set([int(e) for e in number_pattern.findall(" ".join(src))]))
        sent_without_src= remove_citations(sentence)
        statements.append({"text":sent_without_src.strip(), "source": int_src})
    return statements

def split_text_by_citations(text):
    """
        splits text into statement by citations i.e. a statement is a sentence with citation, if a sentence does not contain citation it's appended to the previous one
    """
    # Define a regex pattern to match the citations in various formats
    citation_pattern =r'\[\d+(?:\s*[-,]\s*\d+)*\](?:\[\d+\])*'
    initial_citations_pattern = r'^(' + citation_pattern + r'\s*)+'
    number_pattern = re.compile(r"\d+")
    
    # Split the text into sentences
    sentences = sent_tokenize(text)
    
    statements = []

    sent_without_src = ''
    for sentence in sentences:
        sentence = sentence.strip()
        src = re.findall(citation_pattern, sentence)

        
        initial_citations_match = re.match(initial_citations_pattern, sentence) 
        if initial_citations_match:
            # If the sentence starts with a citation, append the source to previous sentence
            # Extract the matched part and find all citations within it
            initial_citations_text = initial_citations_match.group(0)
            initial_citations = re.findall(citation_pattern, initial_citations_text)
            initial_int_src = list(set([int(e) for e in number_pattern.findall(" ".join(initial_citations))]))
            #print("initial_int_src",initial_int_src)
            sentence = sentence.replace(initial_citations[0], "")
            #print(sentence)
            if sent_without_src:
                statements.append({"text":sent_without_src.strip()[:-1]+" "+ initial_citations[0]+".", "source": initial_int_src})
                sent_without_src = ""
            elif statements:
                statements[-1]["source"].extend(initial_int_src)
                statements[-1]["text"] += ' ' + initial_citations[0]
                # if statements[-1]["source"] == []:
                #     
            for s in initial_citations:
                src.remove(s)
                    
        int_src = list(set([int(e) for e in number_pattern.findall(" ".join(src))]))
        if len(int_src):
            sent_without_src += ' ' + sentence
            statements.append({"text":sent_without_src.strip(), "source": int_src})
            sent_without_src = ""
        else:
            sent_without_src += ' ' + sentence
    
    # Add the last statement
    if sent_without_src.strip(",").strip(".").strip():
        statements.append({"text":sent_without_src.strip(), "source": int_src})
    
    return statements




def get_source_from_text(passage,cited_docids=False):
    if cited_docids:
        pattern = re.compile(r"\[\d+#\d+(?:,\s*\d+#\d+)*\](?:,\s\[\d+#\d+(?:,\s*\d+#\d+)*\])*")
    else:
        pattern = re.compile(r"\[\d+(?:,\s*\d+)*\](?:,\s\[\d+(?:,\s*\d+)*\])*")
    number_pattern = re.compile(r"\d+")
    citations = pattern.findall(passage)
    sources = []
    if cited_docids:
        sources =citations
    else:
        for src in citations:
            int_src = list(set([int(e) for e in number_pattern.findall(src)]))
            sources.extend(int_src)
    return sources


def _run_align_score(passage, claim):
    global alignscore
    score = alignscore.score(contexts=[passage], claims=[claim])

    return score


def _run_nli_autoais(passage, claim):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    global autoais_model, autoais_tokenizer
    init_tokenizers()
    model_dtype = autoais_model.dtype

    input_text = "premise: {} hypothesis: {}".format(passage, claim)
    input_ids = autoais_tokenizer(
        input_text, return_tensors="pt", truncation=True
    ).input_ids.to(autoais_model.device) #.to(model_dtype)

    ##### not compatible with vllm
    # sampling_params = SamplingParams(
    #     max_new_tokens=1024,  # Renamed from max_new_tokens
    #     # vLLM handles padding automatically for batching
    #     # Other common parameters: top_p=0.95, frequency_penalty=0.1
    # )
    # outputs = llm.generate([input_text], sampling_params)
    # result = outputs[0].outputs[0].text
    # inference = 1 if result == "1" else 0

    with torch.inference_mode():
        outputs = autoais_model.generate(input_ids, max_new_tokens=1024)
    result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference = 1 if result == "1" else 0
    return inference


def _run_nli_autoais_with_scores(passage, claim):

    global autoais_model, autoais_tokenizer
    init_tokenizers()

    # Get the token IDs for "0" and "1"
    # Ensure add_special_tokens=False so it just gives the ID for "0" or "1"
    try:
        # For some tokenizers, "0" and "1" might be represented as single tokens.
        # It's crucial to check if your tokenizer tokenizes them as expected.
        token_id_0 = autoais_tokenizer.encode("0", add_special_tokens=False)[0]
        token_id_1 = autoais_tokenizer.encode("1", add_special_tokens=False)[0]
    except IndexError:
        print("Warning: '0' or '1' might not be single tokens in your tokenizer. Adjust logic if needed.")
        # Fallback for demonstration if "0" or "1" are not single tokens, though this is less robust.
        # In a real scenario, you'd need to ensure your model outputs single tokens for labels.
        token_id_0 = None
        token_id_1 = None


    input_text = "premise: {} hypothesis: {}".format(passage, claim)
    input_ids = autoais_tokenizer(
        input_text, return_tensors="pt", truncation=True
    ).input_ids.to(autoais_model.device)

    with torch.inference_mode():
        outputs = autoais_model.generate(
            input_ids,
            max_new_tokens=1,  # We only expect '0' or '1' as output, so only generate 1 token
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,   # Use greedy decoding for a deterministic output and its probability
            num_beams=1        # Ensure greedy search
        )

    # The generated sequence will contain the predicted token (0 or 1)
    # outputs.sequences is typically (batch_size, input_length + num_generated_tokens)
    generated_token_id = outputs.sequences[0, -1].item() # Get the last (and only) generated token ID

    # The scores will contain the logits for the first (and only) generated token
    # outputs.scores is a tuple, where each element corresponds to a generated token.
    # Since max_new_tokens=1, outputs.scores will have one element.
    logits_for_first_token = outputs.scores[0] # Shape: (batch_size, vocab_size)

    # Assuming batch_size = 1, get the logits for the specific token IDs "0" and "1"
    prob_0 = 0.0 # Default if token_id_0 not found
    prob_1 = 0.0 # Default if token_id_1 not found

    if token_id_0 is not None and token_id_1 is not None:
        # Apply softmax to convert logits to probabilities
        probabilities = torch.softmax(logits_for_first_token, dim=-1)

        # Get the probability for token "0" and token "1"
        prob_0 = probabilities[0, token_id_0].item()
        prob_1 = probabilities[0, token_id_1].item()
    else:
        # Fallback if "0" or "1" are not single tokens.
        # In this case, we can only get the probability of the *generated* token.
        print("Could not directly calculate prob for '0' and '1'. Calculating prob for generated token.")
        probabilities = torch.softmax(logits_for_first_token, dim=-1)
        if generated_token_id == token_id_0:
            prob_0 = probabilities[0, generated_token_id].item()
        elif generated_token_id == token_id_1:
            prob_1 = probabilities[0, generated_token_id].item()


    # Determine the predicted label
    result_token = autoais_tokenizer.decode(generated_token_id, skip_special_tokens=True)
    inference = 1 if result_token == "1" else 0 # Assuming "1" for positive, "0" for negative

    # Determine the probability associated with the *predicted* inference label
    predicted_label_probability = 0.0
    if inference == 1:
        predicted_label_probability = prob_1
    else:
        predicted_label_probability = prob_0

    return inference, predicted_label_probability



# Function to extract all cited numbers
def extract_cited_numbers(text):
    # Find all matches of the citation pattern
    citation_pattern = r'\[\d+(?:\s*[-,]\s*\d+)*\](?:\[\d+\])*'
    matches = re.findall(citation_pattern, text)
    
    # Define a pattern to extract numbers from the matches
    number_pattern = re.compile(r'\d+')
    
    # Extract numbers from the matches and convert them to integers
    cited_numbers = []
    for match in matches:
        numbers = number_pattern.findall(match)
        cited_numbers.extend(int(num) for num in numbers)
    
    # Return the list of cited numbers
    return list(set(cited_numbers))

# Function to replace citation numbers with corresponding docid
def replace_citations(text, doc_list):
    def replace_match(match):
        citation_text = match.group(0)
        # Find all numbers within the citation
        numbers = re.findall(r'\d+', citation_text)
        for number in numbers:
            citation_number = int(number) - 1  # Convert to zero-based index
            if 0 <= citation_number < len(doc_list):
                docid = doc_list[citation_number]['docid']
                citation_text = citation_text.replace(f'{number}]', f'{docid}]')
                citation_text = citation_text.replace(f'{number},', f'{docid},')
        return citation_text
    # Use regular expression to find and replace citations
    citation_pattern =r'\[\d+(?:\s*[-,]\s*\d+)*\](?:\[\d+\])*'
    updated_text = re.sub(citation_pattern, replace_match, text)
    return updated_text

########### Tests

def test_reposition_period_after_citation():
    # Input string
    text = 'This is a [4] sentence .[1][4][6] Another sentence. [1,2,3] and a normal sentence [2]. another normal [1] case. [4]'

    # Perform the replacement
    result = reposition_period_after_citation

    print(result)

def test_split_text_by_citations():
    # Example text
    text = "something something something, something. Something something. [1][4][2] some other thing. something [1] something something. Another example with mixed citations [1, 2, 3] and [4][5]. More text."
    print(text)
    # Split the text into statements
    statements = split_text_by_citations(text)

    # Print the resulting statements
    for i, statement in enumerate(statements, 1):
        print(f'Statement {i}: {statement}')
