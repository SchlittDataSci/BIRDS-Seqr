import hashlib
import spacy
import json_repair

import pandas as pd
from lingua import LanguageDetectorBuilder
from itertools import chain
from ast import literal_eval


#########################################
#                                       #
#      TABULAIRITY ENVIRONMENT PREP     #
#                                       #
#########################################


import sys
sys.path.append('../TabulAIrity/src/tabulairity/')
from tabulairity import core as tb


BIRDSNetDf = pd.read_csv('PromptEvalNet.csv')
BIRDSNet = tb.buildChatNet(BIRDSNetDf)
tb.modelRoutes = tb.prepEnvironment()


#########################################
#                                       #
#      SENTENCE TOKENIZERS              #
#                                       #
#########################################


langToModel: dict[str, str] = {
    "CA": "ca_core_news_sm",
    "ZH": "zh_core_web_sm",
    "HR": "hr_core_news_sm",
    "DA": "da_core_news_sm",
    "NL": "nl_core_news_sm",
    "EN": "en_core_web_sm",
    "FI": "fi_core_news_sm",
    "FR": "fr_core_news_sm",
    "DE": "de_core_news_sm",
    "EL": "el_core_news_sm",
    "IT": "it_core_news_sm",
    "JA": "ja_core_news_sm",
    "KO": "ko_core_news_sm",
    "LT": "lt_core_news_sm",
    "MK": "mk_core_news_sm",
    "NB": "nb_core_news_sm",
    "PL": "pl_core_news_sm",
    "PT": "pt_core_news_sm",
    "RO": "ro_core_news_sm",
    "RU": "ru_core_news_sm",
    "SL": "sl_core_news_sm",
    "ES": "es_core_news_sm",
    "SV": "sv_core_news_sm",
    "UK": "uk_core_news_sm",
}

fallbackModel = "xx_sent_ud_sm"
codonMapperDf = pd.read_csv('mappers/BitsToCodons.csv').set_index(['bits','input'])
codonMapper = lambda x,y: codonMapperDf.at[(x,y),'output']
modelCache: dict[str, spacy.language.Language] = {}
detector = LanguageDetectorBuilder.from_all_languages().build()

def loadModel(langCode: str) -> spacy.language.Language:
    """Loads and caches a spaCy model by language code."""
    modelName = langToModel.get(langCode, fallbackModel)

    if modelName not in modelCache:
        try:
            modelCache[modelName] = spacy.load(modelName)
        except OSError:
            raise OSError(f"Model '{modelName}' not installed. Run: python -m spacy download {modelName}")

    return modelCache[modelName]


def getModelMetadata(nlp: spacy.language.Language) -> dict:
    """Extracts version metadata from the model."""
    return {"modelVersion": nlp.meta.get("version", "unknown")}


def parseText(text: str) -> dict:
    """Detects language, tokenizes text, and returns metadata and sentences."""
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    detected = detector.detect_language_of(text)
    langCode = detected.iso_code_639_1.name if detected else "EN"

    nlp = loadModel(langCode)
    modelName = langToModel.get(langCode, fallbackModel)
    
    doc = nlp(text)

    return {
        "metadata": {
            "input hash": hashlib.md5(text.encode("utf-8")).hexdigest(),
            "chatnet hash": hashlib.file_digest(open('PromptEvalNet.csv', 'rb'), 'md5').hexdigest(),
            "language": langCode,
            "spacy version": spacy.__version__,
            "model name": modelName,
            **getModelMetadata(nlp),
        },
        "text": text,
        "sentences": [sent.text.strip() for sent in doc.sents],
        "tokens": list(chain.from_iterable([[token.text for token in sent if not token.is_space] for sent in doc.sents])),
        "reads": dict()
    }
    import json


#########################################
#                                       #
#      STRING ENCODING METHODS          #
#                                       #
#########################################


def strToCodons(s: str) -> int:
    """Hashes a str to semirandom 6 bit number and maps to ATGC codon"""
    h = hashlib.sha256(s.encode("utf-8")).digest()
    n = int.from_bytes(h[:8], "big")
    return codonMapper(6,(n % 64) + 1)


def hashTextsToCodons(parsedText: dict,
                      selectedReads: list):
    """Returns psuedo codon sequences for characters, words, and sentences in parsed text"""
    reads = dict()
    if 'hash by characters' in selectedReads:
        reads['hash by characters'] = ''.join([strToCodons(i) for i in parsedText['text']])
    if 'hash by words' in selectedReads:
        reads['hash by words'] = ''.join([strToCodons(i) for i in parsedText['tokens']])
    if 'hash by sentences' in selectedReads:
        reads['hash by sentences'] = ''.join([strToCodons(i) for i in parsedText['sentences']])

    parsedText['reads'] = parsedText['reads'] | reads


#########################################
#                                       #
#      TABULAIRITY EVAL METHODS         #
#                                       #
#########################################


def cleanListOld(text):
    """LLM output serial object cleaner"""
    if text == '[]':
        return []
    try:
        text = text.replace("```",'')
        if text.startswith('json'):
            text = text[4:]
        while text.startswith('\n'):
            text = text[1:]
        while text.endswith('\n'):
            text = text[:-1]
        text = text.strip()
        text = ''.join(text.split('\n'))
        text = text.replace(' null','None')
        return literal_eval(text)
    except Exception as e:
        print(e,text)


def cleanList(text):

    """LLM output serial object cleaner using json-repair"""
    if not text or str(text).strip() in ('[]', '{}'):
        return []    
    try:
        return json_repair.loads(text)
    except Exception as e:
        print(f"[cleanList] json_repair failed: {e}\nOriginal text: {text}")
        return None


chatFx = {'cleanList': lambda x,y: cleanList(x)}
toDiscard = lambda x: '-' in x or x.endswith('raw') or x.endswith('_prompt')


def evalStrToCodons(parsedText: dict,
                    keepCodons: bool = True,
                    keepReasoning: bool = False,
                    verbosity: int = 0):
    """Uses LLMs to encode the intent of each sentence in a prompt to psuedocodon sequences"""
    sentences = parsedText['sentences']
    sentenceContextsDf = pd.DataFrame({
                            'target_sentence': sentences,
                            'previous_context': [''] + sentences[:-1],
                            'next_context': sentences[1:] + ['']
    })
    
    results = []
    for index,sentenceContext in sentenceContextsDf.iterrows():
        result = tb.walkChatNet(BIRDSNet,
                                varStore = sentenceContext,
                                fxStore = chatFx,
                                runAsync = True,
                                verbosity = verbosity)
        result = {key:value for key,value in result.items() if not toDiscard(key)}
        results.append(result)

    if keepCodons:
        codons = ""
        for result in results:
            codons += codonMapper(2,result['sentence_type']['id']+1)
            codons += codonMapper(4,result['attack_type']['id']+1)
        parsedText['reads']['Eval by sentences'] = codons

    if keepReasoning:
        reasoning = []
        reasoning.append({
            'sentence type id': result['sentence_type']['id'],
            'sentence type reasoning': result['sentence_type']['reasoning'],
            'attack type id': result['attack_type']['id'],
            'attack type reasoning': result['attack_type']['reasoning']
        })
        parsedText['threat reasoning'] = reasoning
        

#########################################
#                                       #
#      TEXT ENCODING METHOD             #
#                                       #
#########################################


def wrapSequence(sequence: str,
                 width: int = 60):
    """Wraps a sequence str to fixed-width lines"""
    return '\n'.join(
        sequence[i:i+width]
        for i in range(0, len(sequence), width)
    )


def generateFASTA(parsedText: dict):
    """Combines reads into a singular FASTA file str"""
    FASTAText = []
    for read,sequence in parsedText['reads'].items():
        read = read.replace(' ','_')
        sequence = ''.join(sequence.split())
        FASTAText.append(f">{read}")
        FASTAText.append(wrapSequence(sequence))
    
    return '\n'.join(FASTAText) + '\n'


def sequenceText(text: str,
                name: str,
                description: str = '',
                useHashes: bool = True,
                useEvals: bool = True,
                keepEvalCodons: bool = True,
                keepEvalReasoning: bool = False,
                keepParsed: bool = False,
                selectedReads: list = ['hash by words','hash by sentences'],
                verbosity: int = 0):
    """Converts input text to sequence representation"""
    parsedText = parseText(text)
    parsedText['name'] = name
    parsedText['description'] = description
    
    if useHashes:
        hashTextsToCodons(parsedText,
                         selectedReads)

    if useEvals:
        evalStrToCodons(parsedText,
                        keepEvalCodons,
                        keepEvalReasoning,
                        verbosity)

    if not keepParsed:
        del parsedText['sentences']
        del parsedText['tokens']

    parsedText['fasta'] = generateFASTA(parsedText)
    

    return parsedText