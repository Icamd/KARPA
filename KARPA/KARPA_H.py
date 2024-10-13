from tqdm import tqdm
import argparse
from utils import *
import random
from client import *
from freebase_func import *
import networkx as nx
import datasets
import os
import heapq
os.environ.pop("http_proxy", None)
os.environ.pop("all_proxy", None)
os.environ.pop("https_proxy", None)
from openai import OpenAI
from prompt_list import *
from evaluate_results import eval_result

def build_graph(graph: list) -> nx.Graph:
    G = nx.Graph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h, t, relation=r.strip())
    return G

def dfs(graph, current_node, depth, path, all_paths, visited_edges):
    if len(path) == depth + 1:
        all_paths.append(path.copy())
        return

    for neighbor in graph.neighbors(current_node):
        edge = (current_node, neighbor)
        if edge not in visited_edges and (neighbor, current_node) not in visited_edges and current_node != neighbor:
            visited_edges.add(edge)
            path.append(neighbor)
            dfs(graph, neighbor, depth, path, all_paths, visited_edges)
            path.pop()
            visited_edges.remove(edge)

def find_paths_of_depth(graph, start_node, depth):
    all_paths = []
    path = [start_node]
    visited_edges = set()
    dfs(graph, start_node, depth, path, all_paths, visited_edges)
    return all_paths

def merge_rule_result(qa_dataset, rule_dataset, n_proc=1, filter_empty=False):
    question_to_rule = dict()
    for data in rule_dataset:
        qid = data["id"]
        predicted_paths = data["prediction"]
        ground_paths = data["ground_paths"]
        question_to_rule[qid] = {
            "predicted_paths": predicted_paths,
            "ground_paths": ground_paths,
        }

    def find_rule(sample):
        qid = sample["id"]
        sample["predicted_paths"] = []
        sample["ground_paths"] = []
        sample["predicted_paths"] = question_to_rule[qid]["predicted_paths"]
        sample["ground_paths"] = question_to_rule[qid]["ground_paths"]
        return sample

    qa_dataset = qa_dataset.map(find_rule, num_proc=n_proc)
    if filter_empty:
        qa_dataset = qa_dataset.filter(
            lambda x: len(x["ground_paths"]) > 0, num_proc=n_proc
        )
    return qa_dataset

def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(results["id"])
        fout = open(path, "a")
        return fout, processed_results

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def dijkstra(graph, start, len_pred, relation_mapping):
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    relations_mapping = {node: [] for node in graph}
    priority_queue = [(0, start, 0, [])]

    visited = set()

    while priority_queue:
        current_total_cost, current_node, edges_count, current_route = heapq.heappop(priority_queue)

        if edges_count > 0:
            current_average_cost = current_total_cost / edges_count
        else:
            current_average_cost = 0

        # 如果该节点已访问过，则跳过
        if current_node in visited:
            continue
        visited.add(current_node)

        for neighbor, weight in graph[current_node].items():
            new_total_cost = current_total_cost + weight
            new_edges_count = edges_count + 1
            new_average_cost = new_total_cost / new_edges_count

            if new_average_cost < distances[neighbor] and new_edges_count <= len_pred:
                distances[neighbor] = new_average_cost
                new_current_route = current_route.copy()
                new_current_route.append(relation_mapping[current_node][neighbor]['relation'])
                heapq.heappush(priority_queue, (new_total_cost, neighbor, new_edges_count, new_current_route))
                relations_mapping[neighbor] = new_current_route

    return distances, relations_mapping


def a_star_3(graph, start, len_pred, h_value_list, relation_mapping):
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    relations_mapping = {node: [] for node in graph}
    priorities = {node: float('infinity') for node in graph}
    priority_queue = [(0, 0, start, 0, [], [])]
    came_from = {start: None}

    visited = set()

    while priority_queue:
        current_priority, current_total_distance, current_node, edge_count, current_route, node_path = heapq.heappop(priority_queue)

        if edge_count >= len_pred:
            continue
        if current_node in visited:
            continue
        visited.add(current_node)

        for neighbor, weight in graph[current_node].items():
            new_total_distance = current_total_distance + weight
            new_edge_count = edge_count + 1
            new_average_distance = new_total_distance / new_edge_count

            if new_average_distance < distances[neighbor] and new_edge_count <= len_pred:
                distances[neighbor] = new_average_distance
                new_current_route = current_route.copy()
                new_current_route.append(relation_mapping[current_node][neighbor]['relation'])
                current_node_path_head = node_path.copy()
                current_node_path_head.append(current_node)
                current_node_path_tail = current_node_path_head.copy()
                current_node_path_tail.append(neighbor)

                priority = new_average_distance + h_value_list[' -> '.join(current_node_path_tail)]
                heapq.heappush(priority_queue, (priority, new_total_distance, neighbor, new_edge_count, new_current_route, current_node_path_head))
                came_from[neighbor] = current_node
                came_from[neighbor] = current_node
                relations_mapping[neighbor] = new_current_route
                priorities[neighbor] = priority

    return distances, relations_mapping, priorities

def select_unique_paths(data_dict, top_n):
    sorted_items = sorted(data_dict.items(), key=lambda item: item[1]['value'])

    unique_paths = []
    seen_paths = set()

    for key, sub_dict in sorted_items:
        path = tuple(sub_dict['path'])
        if path not in seen_paths:
            unique_paths.append((path, sub_dict['value']))
            seen_paths.add(path)
        if len(unique_paths) >= top_n:
            break

    return unique_paths


def top_three_indices(lst, width):
    sorted_indices = sorted(range(len(lst)), key=lambda i: lst[i], reverse=False)

    return sorted_indices[:width]

def find_path(backtrace_dict, start_tail_entity):
    path = []
    current_entity = start_tail_entity

    while current_entity in backtrace_dict:
        head_entity, relation = backtrace_dict[current_entity]
        path.append((head_entity, relation, current_entity))
        current_entity = head_entity

    path.reverse()
    return path

def extract_text_segment(text):
    last_period_index = text.rfind('.')

    if last_period_index == -1:
        return ''

    segment_start = text.rfind('.', 0, last_period_index)
    if segment_start == -1:
        return text
    else:
        return text[segment_start + 1:last_period_index + 1].strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="webqsp", help="choose the dataset.")
    parser.add_argument("--max_length", type=int,
                        default=2560, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.4, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float,
                        default=0, help="the temperature in reasoning stage.")
    parser.add_argument("--LLM_type", type=str,
                        default="gpt-4o-mini", help="base LLM model.")
    parser.add_argument("--opeani_api_keys", type=str,
                        default="", help="Add your own api keys for LLM inference.")
    parser.add_argument("--LLM_URL", type=str,
                        default="https://api.openai.com/v1", help="Add the URL for the api key.")
    parser.add_argument("--output_file", type=str,
                        default="output.txt", help="None")
    args = parser.parse_args()
    if args.dataset == 'cwq':
        args.rule_path = '../data/ground_cwq.jsonl'

        datas, question_string = prepare_dataset(args.dataset)
        dt = datasets.load_dataset('rmanluo/RoG-cwq', split="test")
        rule_dataset = load_jsonl(args.rule_path)
        dt = merge_rule_result(dt, rule_dataset)
        output_file_h = "../predictions/cwq/karpa_h/predictions.jsonl"
        fout_h, processed_list_h = get_output_file(output_file_h, force=False)
        id_str = 'ID'
    elif args.dataset == 'webqsp':
        args.rule_path = './data/ground_webqsp.jsonl'
        datas, question_string = prepare_dataset(args.dataset)
        dt = datasets.load_dataset('rmanluo/RoG-webqsp', split="test")
        rule_dataset = load_jsonl(args.rule_path)
        dt = merge_rule_result(dt, rule_dataset)
        output_file_h = "../predictions/webqsp/karpa_h/predictions.jsonl"
        fout_h, processed_list_h = get_output_file(output_file_h, force=False)
        id_str = 'QuestionId'

    client = OpenAI(
        api_key=args.opeani_api_keys,
        base_url=args.LLM_URL,
    )

    total = 0
    datasets = dt.shuffle(seed=42)
    for dataset in tqdm(datasets):
        id = dataset['id']
        if id in processed_list_h:
            continue
        for data in datas:
            if data[id_str]==id:
                break
        question = data[question_string]
        topic_entity = {}
        best_predict_paths = {}
        for key, value in data['topic_entity'].items():
            topic_entity[value] = {0: [value]}
            best_predict_paths[value] = []
        trace_back = {}
        cluster_chain_of_entities = []
        pre_relations = []
        pre_heads= [-1] * len(topic_entity)
        flag_printed = False
        graph = build_graph(dataset['graph'])

        skip_current_sample = False
        all_node_list = list(graph.nodes())
        for answer in dataset['answer']:
            if answer not in all_node_list:
                skip_current_sample = True
                break
        if skip_current_sample == True:
            continue

        if total == 300:
            break
        total += 1

        all_edges = list(graph.edges())
        all_edges_list = []
        all_edges_nodes_list = []
        for u,v in all_edges:
            if graph.has_edge(u, v) and graph[u][v]['relation'] not in all_edges_list:
                all_edges_list.append(graph[u][v]['relation'])
                all_edges_nodes_list.append([u, v])

        all_keys = list(topic_entity.keys())
        concat_keys = ', '.join(all_keys)
        print('start openai')
        response = client.chat.completions.create(model=args.LLM_type, messages=[
            {"role": "system", "content": "You are an AI assistant that helps people find information."},
            {"role": "user", "content": prompt1 + question + "\n" + "Topic Entity: " + concat_keys + "\nA:"}], max_tokens=1000, temperature=0.1, stream=False)
        relative_parts = response.choices[0].message.content
        print('end openai')

        reply_relations = re.findall(r'\{([^}]*)\}', relative_parts)
        relations_list = []
        for relation in reply_relations:
            relations_list.extend([r.strip() for r in relation.split(',') if r.strip()])
        relations_list.append(question)

        all_relations_embeddings = []
        all_relations_embeddings.append(sentence_emb(all_edges_list))
        cosine_scores = util.pytorch_cos_sim(sentence_emb(relations_list), all_relations_embeddings[0])
        clip_num = 30//len(relations_list)
        choosen_set = []
        for i in range(len(relations_list)):
            cost = (1 - cosine_scores[i]).tolist()
            indexs = top_three_indices(cost, clip_num)
            for index in indexs:
                if all_edges_list[index] not in choosen_set:
                    choosen_set.append(all_edges_list[index])

        concat_relations = '; '.join(choosen_set)
        print('start openai')
        response = client.chat.completions.create(model=args.LLM_type, messages=[
            {"role": "system", "content": "You are an AI assistant that helps people find information."},
            {"role": "user", "content": prompt2 + question + "\nTopic Entity: " + concat_keys + "\nRelations: " + concat_relations + "\nA:"}], max_tokens=1000, temperature=0.1, stream=False)
        relative_parts = response.choices[0].message.content
        print('end openai')

        reply_relations1 = re.findall(r'\{([^}]*)\}', relative_parts)
        temp = []
        for reply_relation in reply_relations1:
            if reply_relation not in temp and reply_relation != '':
                temp.append(reply_relation)
        reply_relations2 = []
        for t in temp:
            reply_relations2.append(t.split(','))

        max_len = 0
        for i in reply_relations2:
            if max_len < len(i):
                max_len = len(i)
        new_result = ''
        entity_dict = {}
        for entity in topic_entity:
            entity_dict[entity] = {}
            grouped_dict = {}
            for i in range(max_len+1):
                grouped_dict[i] = []
            for i in range(max_len+1):
                paths = find_paths_of_depth(graph, entity, i)
                for path in paths:
                    if path[i] not in grouped_dict[i]:
                        grouped_dict[i].append(path[i])
            entity_dict[entity]['grouped_dict'] = grouped_dict
            sorted_keys = sorted(grouped_dict.items(), key=lambda x: x[0], reverse=False)
            grouped_keys = [keys for value, keys in sorted_keys]
            relations_embeddings = []
            entity_lists = []
            for i in range(len(grouped_keys) - 1):
                k_hop_relation_list = []
                entity_list = []
                for u in grouped_keys[i]:
                    for v in grouped_keys[i + 1]:
                        if u != v and graph.has_edge(u, v):
                            k_hop_relation_list.append(graph[u][v]['relation'])
                            entity_list.append([u, v])
                relations_embeddings.append(sentence_emb(k_hop_relation_list))
                entity_lists.append(entity_list)
            entity_dict[entity]['relations_embeddings'] = relations_embeddings
            entity_dict[entity]['entity_lists'] = entity_lists
            entity_dict[entity]['relation_cost'] = []
            entity_dict[entity]['all_entity_pair'] = []
        new_cluster_chain_of_entities = []
        new_cluster_chain_of_entities_backup = []
        new_cluster_chain_of_entities_beam = []
        new_cluster_chain_of_entities_backup_beam = []
        for reasoning_path in reply_relations2:
            for entity in topic_entity:
                for i in range(len(reasoning_path)):
                    cosine_scores = util.pytorch_cos_sim(sentence_emb([reasoning_path[i]]), entity_dict[entity]['relations_embeddings'][i])
                    entity_dict[entity]['relation_cost'].extend((1 - cosine_scores[0]).tolist())
                    entity_dict[entity]['all_entity_pair'].extend(entity_dict[entity]['entity_lists'][i])
                    i = i + 1
            for entity in topic_entity:
                dijkstra_graph = {}
                all_paths = {}
                all_paths_beam = {}
                for i in range(len(entity_dict[entity]['all_entity_pair'])):
                    node1, node2 = entity_dict[entity]['all_entity_pair'][i]
                    cost = entity_dict[entity]['relation_cost'][i]

                    if node1 not in dijkstra_graph:
                        dijkstra_graph[node1] = {}
                    if node2 not in dijkstra_graph:
                        dijkstra_graph[node2] = {}

                    dijkstra_graph[node1][node2] = cost

                depth = len(reasoning_path)
                shortest_paths, relations_mapping = dijkstra(dijkstra_graph, entity, depth, graph)
                for key, value in shortest_paths.items():
                    if len(relations_mapping[key]) == depth:
                        if key not in all_paths:
                            if value > 0.001:
                                all_paths[key] = {'value': value, 'path': relations_mapping[key]}
                            else:
                                all_paths[key] = {'value': 0.0, 'path': relations_mapping[key]}
                        else:
                            if value < all_paths[key]['value']:
                                all_paths[key] = {'value': value, 'path': relations_mapping[key]}
                dict_temp = dijkstra_graph[entity]
                first_adj_relations = []
                list_temp = sorted(dict_temp, key=dict_temp.get)[:16]
                for adj_rela in list_temp:
                    first_adj_relations.append(graph[entity][adj_rela]['relation'])
                for key, value in shortest_paths.items():
                    if len(relations_mapping[key]) == depth and relations_mapping[key][0] in first_adj_relations:
                        if key not in all_paths_beam:
                            if value > 0.001:
                                all_paths_beam[key] = {'value': value, 'path': relations_mapping[key]}
                            else:
                                all_paths_beam[key] = {'value': 0.0, 'path': relations_mapping[key]}
                        else:
                            if value < all_paths_beam[key]['value']:
                                all_paths_beam[key] = {'value': value, 'path': relations_mapping[key]}
                num_path = 16
                related_paths = select_unique_paths(all_paths, num_path)
                related_paths_beam = select_unique_paths(all_paths_beam, num_path)
                for related_path in related_paths[:8]:
                    keys = []
                    for key, value in relations_mapping.items():
                        if value == list(related_path[0]):
                            keys.append(key)
                    new_cluster_chain_of_entities.append((entity, ' -> '.join(related_path[0]), ', '.join(keys)))
                for related_path in related_paths[8:]:
                    keys = []
                    for key, value in relations_mapping.items():
                        if value == list(related_path[0]):
                            keys.append(key)
                    new_cluster_chain_of_entities_backup.append((entity, ' -> '.join(related_path[0]), ', '.join(keys)))

                for related_path in related_paths_beam[:8]:
                    keys = []
                    for key, value in relations_mapping.items():
                        if value == list(related_path[0]):
                            keys.append(key)
                    new_cluster_chain_of_entities_beam.append((entity, ' -> '.join(related_path[0]), ', '.join(keys)))
                for related_path in related_paths_beam[8:]:
                    keys = []
                    for key, value in relations_mapping.items():
                        if value == list(related_path[0]):
                            keys.append(key)
                    new_cluster_chain_of_entities_backup_beam.append((entity, ' -> '.join(related_path[0]), ', '.join(keys)))

        extract_answers = []
        if len(new_cluster_chain_of_entities) == 0:
            extract_answers = []
        else:
            h_value_dicts = {}
            path_batch = 3
            lens = []
            for i in reply_relations2:
                if len(i) not in lens:
                    lens.append(len(i))
            for entity in topic_entity:
                h_value_dicts[entity] = {}
                for length in lens:
                    h_value_dicts[entity][length] = {}
                    for length1 in lens:
                        h_value_dicts[entity][length][length1]= []

            for entity in topic_entity:
                for length in lens:
                    key_list = []
                    path_dict = {}
                    paths = find_paths_of_depth(graph, entity, length)
                    for path in paths:
                        key = []
                        for i in range(len(path) - 1):
                            current_element = path[i]
                            next_element = path[i + 1]
                            key.append(graph[current_element][next_element]['relation'])

                        key_str = " -> ".join(key)
                        if key_str in path_dict:
                            path_dict[key_str].append(path[-1])
                        else:
                            path_dict[key_str] = [path[-1]]
                    key_list = list(path_dict.keys())
                    embeddings = sentence_emb(key_list)
                    for related_path in reply_relations2:
                        cosine_scores = util.pytorch_cos_sim(sentence_emb([" -> ".join(related_path)]), embeddings)
                        choosen_set = []
                        choosen_set_keys = []
                        cost = []
                        cost = (1 - cosine_scores[0]).tolist()
                        indexs = top_three_indices(cost, len(cost))
                        for index in indexs:
                            if key_list[index] not in choosen_set:
                                choosen_set.append(key_list[index])
                                choosen_set_keys.append(path_dict[key_list[index]])
                        h_value_dicts[entity][len(related_path)][length].append(choosen_set)
                        h_value_dicts[entity][len(related_path)][length].append(choosen_set_keys)

            counter = 0
            all_answers = []
            extract_answers = []
            results = ''
            while True:
                for entity in topic_entity:
                    for length1 in lens:
                        temp = []
                        for length2 in lens:
                            for i in range(counter, counter + 8):
                                if i < len(h_value_dicts[entity][length1][length2][0]):
                                    temp.append((entity, h_value_dicts[entity][length1][length2][0][i], ', '.join(h_value_dicts[entity][length1][length2][1][i])))
                        result = reasoning_h4(question, [[temp]], args)
                        results += result
                        entities = re.findall(r'\{([^}]*)\}', result)
                        for i in entities:
                            all_answers.extend(i.split(', '))
                        final_entities2 = [[x[2].split(', ') for x in sublist] for sublist in [[temp]][-1]][0]
                        final_entities = []
                        for final in final_entities2:
                            final_entities.extend(final)
                        for i in all_answers:
                            if i in final_entities:
                                extract_answers.append(i)

                if counter >= 8 and len(extract_answers) != 0:
                    break
                if counter >= 8:
                    break
                counter += 8

        extract_answers = list(set(extract_answers))
        if len(extract_answers) == 0:
            results = generate_without_explored_paths(question, args)
            use_exhaustivity = True

        if use_exhaustivity == True:
            for node in graph.nodes():
                if node.lower() in results.lower():
                    extract_answers.append(node)

        use_exhaustivity = False
        extract_answers = list(set(extract_answers))
        for key in topic_entity:
            if key in extract_answers:
                extract_answers.remove(key)

        answer = '\n'.join(extract_answers)

        format_result = {
            "id": id,
            "question": question,
            "prediction": answer,
            "ground_truth": dataset["answer"],
            "results": results,
            "reasoning_chains": cluster_chain_of_entities
        }
        fout_h.write(json.dumps(format_result) + "\n")
        fout_h.flush()


    fout_h.close()
    eval_result(output_file_h)
