from SPARQLWrapper import SPARQLWrapper, JSON
from utils import *

SPARQLPATH = "http://192.168.80.12:8890/sparql"

# pre-defined sparqls
sparql_head_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ns:%s ?relation ?x .\n}"""
sparql_tail_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ?x ?relation ns:%s .\n}"""
sparql_tail_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\nns:%s ns:%s ?tailEntity .\n}""" 
sparql_head_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\n?tailEntity ns:%s ns:%s  .\n}"""
sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  {\n    ?entity ns:type.object.name ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n  UNION\n  {\n    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n}"""
    
def check_end_word(s):
    words = [" ID", " code", " number", "instance of", "website", "URL", "inception", "image", " rate", " count"]
    return any(s.endswith(word) for word in words)

def abandon_rels(relation):
    if relation == "type.object.type" or relation == "type.object.name" or relation.startswith("common.") or relation.startswith("freebase.") or "sameAs" in relation:
        return True


def execurte_sparql(sparql_query):
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results["results"]["bindings"]


def replace_relation_prefix(relations):
    return [relation['relation']['value'].replace("http://rdf.freebase.com/ns/","") for relation in relations]

def replace_entities_prefix(entities):
    return [entity['tailEntity']['value'].replace("http://rdf.freebase.com/ns/","") for entity in entities]


def id2entity_name_or_type(entity_id):
    sparql_query = sparql_id % (entity_id, entity_id)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if len(results["results"]["bindings"])==0:
        return "UnName_Entity"
    else:
        return results["results"]["bindings"][0]['tailEntity']['value']

from freebase_func import *
from prompt_list import *
import json
import time
import openai
import re
from prompt_list import *
from sentence_transformers import util
from sentence_transformers import SentenceTransformer

Sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device='cpu')
def sentence_emb(sentence_list):
    return Sentence_model.encode(sentence_list)

def clean_relations(string, entity_id, head_relations):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
    if not relations:
        return False, "No relations found"
    return True, relations


def if_all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)


def clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations):
    relations = []
    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    i=0
    for relation in topn_relations:
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": False})
        i+=1
    return True, relations


def construct_relation_prune_prompt(question, entity_name, total_relations, args):
    return extract_relation_prompt % (args.width, args.width) + question + '\nTopic Entity: ' + entity_name + '\nRelations: '+ '; '.join(total_relations) + "\nA: "
        

def construct_entity_score_prompt(question, relation, entity_candidates):
    return score_entity_candidates_prompt.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '


def relation_search_prune(entity_id, entity_name, pre_relations, pre_head, question, args, graph, if_score_match):
    head_relations = []
    for neighbor, _, data in graph.edges(entity_name, data=True):
        head_relations.append(data.get('relation'))

    if args.remove_unnecessary_rel:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]

    if pre_head:
        for x in pre_relations:
            if x in head_relations:
                head_relations.remove(x)
    if len(head_relations) == 0:
        return []
    head_relations = list(set(head_relations))
    total_relations = head_relations.copy()
    total_relations.sort()
    
    if args.prune_tools == "llm":
        prompt = construct_relation_prune_prompt(question, entity_name, total_relations, args)

        result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
        flag, retrieve_relations_with_scores = clean_relations(result, entity_id, head_relations)
        for retrieve_relation in retrieve_relations_with_scores:
            if retrieve_relation["head"] == False:
                if if_score_match == False:
                    retrieve_relation["score"] = 0.0
                    retrieve_relation["head"] = True
                else:
                    embeddings = Sentence_model.encode(head_relations)
                    cosine_scores = util.pytorch_cos_sim(Sentence_model.encode([retrieve_relation["relation"]]), embeddings)
                    score_list = cosine_scores[0].tolist()
                    for r in range(len(score_list)):
                        max_index = score_list.index(max(score_list))
                        relations = [item['relation'] for item in retrieve_relations_with_scores]
                        if head_relations[max_index] in relations:
                            score_list[max_index] = 0
                        else:
                            retrieve_relation["relation"] = head_relations[max_index]
                            retrieve_relation["head"] = True
                            retrieve_relation["score"] = score_list[max_index]*retrieve_relation["score"]
                            break
                    if retrieve_relation["head"] == False:
                        retrieve_relation["score"] = 0
                        retrieve_relation["head"] = True

    else:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        topn_relations, topn_scores = retrieve_top_docs(question, total_relations, model, args.width)
        flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations) 

    if flag:
        return retrieve_relations_with_scores
    else:
        return []


def relation_search_prune_dijkstra(entity_id, entity_names, pre_relations, pre_head, question, args, graph, if_score_match):
    head_relations = {}
    total_relations = []
    for entity_name in entity_names:
        head_relations[entity_name] = []
        for neighbor, _, data in graph.edges(entity_name, data=True):
            head_relations[entity_name].append(data.get('relation'))

        if args.remove_unnecessary_rel:
            head_relations[entity_name] = [relation for relation in head_relations[entity_name] if not abandon_rels(relation)]
        if pre_head:
            for x in pre_relations:
                if x in head_relations[entity_name]:
                    head_relations[entity_name].remove(x)
        head_relations[entity_name] = list(set(head_relations[entity_name]))
        total_relations.extend(head_relations[entity_name])

    if len(total_relations) == 0:
       return [], None, None
    total_relations = list(set(total_relations))
    total_relations.sort()

    if args.prune_tools == "llm":
        prompt = construct_relation_prune_prompt(question, entity_id, total_relations, args)

        result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
        flag, retrieve_relations_with_scores = clean_relations(result, entity_id, total_relations)
        for retrieve_relation in retrieve_relations_with_scores:
            if retrieve_relation["head"] == False:
                if if_score_match == False:
                    retrieve_relation["score"] = 0.0
                    retrieve_relation["head"] = True
                else:
                    embeddings = Sentence_model.encode(total_relations)
                    cosine_scores = util.pytorch_cos_sim(Sentence_model.encode([retrieve_relation["relation"]]), embeddings)
                    score_list = cosine_scores[0].tolist()
                    for r in range(len(score_list)):
                        max_index = score_list.index(max(score_list))
                        relations = [item['relation'] for item in retrieve_relations_with_scores]
                        if total_relations[max_index] in relations:
                            score_list[max_index] = 0
                        else:
                            retrieve_relation["relation"] = total_relations[max_index]
                            retrieve_relation["head"] = True
                            retrieve_relation["score"] = score_list[max_index] * retrieve_relation["score"]
                            break
                    if retrieve_relation["head"] == False:
                        retrieve_relation["score"] = 0
                        retrieve_relation["head"] = True

    else:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        topn_relations, topn_scores = retrieve_top_docs(question, total_relations, model, args.width)
        flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations)

    if flag:
        return retrieve_relations_with_scores, head_relations, total_relations
    else:
        return []

    
def entity_search(entity, relation, graph, nodes_allowed, head=True):
    if head:
        entity_ids = [neighbor for neighbor in graph[entity] if graph[entity][neighbor].get('relation') == relation]
    else:
        head_entities_extract = sparql_head_entities_extract% (entity, relation)
        entities = execurte_sparql(head_entities_extract)

    new_entity = [entity for entity in entity_ids]
    new_entity = list(set(new_entity))
    filter_new_entity = [item for item in new_entity if item in nodes_allowed]
    return filter_new_entity

def entity_score(question, entity_candidates_id, score, relation, args):
    entity_candidates = entity_candidates_id.copy()
    if all_unknown_entity(entity_candidates):
        return [1 / len(entity_candidates) * score] * len(entity_candidates), entity_candidates, entity_candidates_id
    entity_candidates = del_unknown_entity(entity_candidates)
    if len(entity_candidates) == 1:
        return [score], entity_candidates, entity_candidates_id
    if len(entity_candidates) == 0:
        return [0.0], entity_candidates, entity_candidates_id

    if args.prune_tools == "llm":
        prompt = construct_entity_score_prompt(question, relation, entity_candidates)#, score)

        result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
        return [float(x) * score for x in
                clean_scores(result, entity_candidates)], entity_candidates, entity_candidates_id

    else:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        topn_entities, topn_scores = retrieve_top_docs(question, entity_candidates, model, args.width)
    if if_all_zero(topn_scores):
        topn_scores = [float(1 / len(topn_scores))] * len(topn_scores)
    return [float(x) * score for x in topn_scores], topn_entities, entity_candidates_id
    
def update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head):
    if len(entity_candidates) == 0:
        entity_candidates.append("[FINISH]")
        entity_candidates_id = ["[FINISH_ID]"]
    candidates_relation = [entity['relation']] * len(entity_candidates)
    topic_entities = [entity['entity']] * len(entity_candidates)
    head_num = [entity['head']] * len(entity_candidates)
    total_candidates.extend(entity_candidates)
    total_scores.extend(scores)
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)
    return total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head


def half_stop(question, cluster_chain_of_entities, depth, args):
    print("No new knowledge added during search depth %d, stop searching." % depth)
    answer = generate_answer(question, cluster_chain_of_entities, args)
    save_2_jsonl(question, answer, cluster_chain_of_entities, file_name=args.dataset)


def generate_answer(question, cluster_chain_of_entities, args): 
    prompt = answer_prompt + question + '\n'
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '
    result = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    return result


def entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args):
    zipped = list(zip(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores))
    sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)
    sorted_entities_id, sorted_relations, sorted_candidates, sorted_topic_entities, sorted_head, sorted_scores = [x[0] for x in sorted_zipped], [x[1] for x in sorted_zipped], [x[2] for x in sorted_zipped], [x[3] for x in sorted_zipped], [x[4] for x in sorted_zipped], [x[5] for x in sorted_zipped]

    entities_id, relations, candidates, topics, heads, scores = sorted_entities_id[:args.width], sorted_relations[:args.width], sorted_candidates[:args.width], sorted_topic_entities[:args.width], sorted_head[:args.width], sorted_scores[:args.width]
    merged_list = list(zip(entities_id, relations, candidates, topics, heads, scores))
    filtered_list = [(id, rel, ent, top, hea, score) for id, rel, ent, top, hea, score in merged_list if score != 0]
    if len(filtered_list) ==0:
        return False, [], [], [], []
    entities_id, relations, candidates, tops, heads, scores = map(list, zip(*filtered_list))

    cluster_chain_of_entities = [[(tops[i], relations[i], candidates[i]) for i in range(len(candidates))]]
    return True, cluster_chain_of_entities, entities_id, relations, heads


def reasoning(question, cluster_chain_of_entities, args):
    prompt = prompt_evaluate + question
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain if x[2][1] != '.']) for sublist in cluster_chain_of_entities for chain in sublist])
    if len(chain_prompt) == 0:
        return False, ''
    prompt += "\nKnowledge Triplets: " + chain_prompt + '\nA: '

    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    
    result = extract_answer(response)
    if if_true(result):
        return True, response
    else:
        return False, response

def reasoning_h4(question, cluster_chain_of_entities, args):
    prompt = prompt_evaluate_h4 + question
    chain_prompt = '\n'.join(
        ['\n'.join(str(x) for x in chain if x[2][1] != '.') for sublist in cluster_chain_of_entities for chain in sublist])
    if len(chain_prompt) == 0:
        return ''
    prompt += "\nReasoning Paths: " + chain_prompt + '\nA: '

    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)

    return response