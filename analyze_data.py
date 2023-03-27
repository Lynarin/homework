#!/usr/bin/env python3
import argparse
import json
from collections import OrderedDict, defaultdict
from statistics import mean

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


class TaskData:
    def __init__(self, project_id, input_image, answer, cant_solve, corrupt_data, duration_ms):
        self.project_id = project_id
        self.input_image = input_image
        self.answer = answer
        self.cant_solve = cant_solve
        self.corrupt_data = corrupt_data
        self.duration_ms = duration_ms


def get_image_name_from_url(url_text):
    return url_text.split('/')[-1].replace('.jpg', '')


def restructure_annotator_data(annotator_data_raw):
    annotator_data_restructured = defaultdict(list)
    id_to_vendor_id = {}
    for key, result_block in annotator_data_raw["results"]["root_node"]["results"].items():
        for result in result_block['results']:
            annotator_user_id = result['user']['id']
            task_project_id = result['project_node_input_id']
            task_input_image = get_image_name_from_url(result['root_input']['image_url'])
            task_answer = result['task_output']['answer']
            task_cant_solve = result['task_output']['cant_solve']
            task_corrupt_data = result['task_output']['corrupt_data']
            task_duration_ms = result['task_output']['duration_ms']

            id_to_vendor_id[annotator_user_id] = result['user']['vendor_user_id']
            annotator_data_restructured[annotator_user_id].append(TaskData(task_project_id, task_input_image, task_answer,
                                                              task_cant_solve, task_corrupt_data, task_duration_ms))

    return annotator_data_restructured, id_to_vendor_id


def get_annotation_durations_statistics(annotators_data):
    result = {}
    for user_id, task_data in annotators_data.items():
        # filter out negative results, as they seem wrong, and they corrupt statistics
        # these results should probably be logged for further analysis
        annonation_times = [int(task.duration_ms) for task in task_data if int(task.duration_ms) > 0]

        result[user_id] = {
            "min": min(annonation_times),
            "max": max(annonation_times),
            "avg": mean(annonation_times),
        }
    return result


def get_annotated_results_counts(annotators_data):
    return {idx: len(res) for idx, res in annotators_data.items()}


def get_difficult_images(answers_per_image):
    result = []
    for image, answers in answers_per_image.items():
        no_num = answers.count("no")
        yes_num = answers.count("yes")
        # an image is difficult if the at least 1/3 of the annotators disagree
        threshold = (no_num + yes_num) / 3
        if no_num > threshold and yes_num > threshold:
            result.append(image)
    return result


def get_annotator_answers_per_image(annotators_data):
    result = defaultdict(list)
    for user_id, task_data in annotators_data.items():
        for task in task_data:
            result[task.input_image].append(task.answer)
    return result


def get_corruption_cannot_solve_statistic(annotators_data):
    statistic_per_annotator = defaultdict(lambda: {"cant_solve": 0, "corrupt_data": 0})
    statistic_per_project = defaultdict(lambda: {"cant_solve": 0, "corrupt_data": 0})
    cant_solve_count = 0
    corrupt_count = 0
    for user_id, task_data in annotators_data.items():
        for task in task_data:
            if task.cant_solve is True:
                statistic_per_annotator[user_id]["cant_solve"] += 1
                statistic_per_project[task.project_id]["cant_solve"] += 1
                cant_solve_count += 1
            elif task.corrupt_data is True:
                statistic_per_annotator[user_id]["corrupt_data"] += 1
                statistic_per_project[task.project_id]["corrupt_data"] += 1
                corrupt_count += 1
    return statistic_per_annotator, statistic_per_project, corrupt_count, cant_solve_count


def convert_to_correct_answers_dict(reference_dataset):
    result = {}
    for image, output in reference_dataset.items():
        result[image] = "yes" if output['is_bicycle'] else "no"
    return result


def calculate_annotator_scores(annotators_tasks, correct_answers):
    stat_per_annotator = {}
    for user_id, task_data in annotators_tasks.items():
        correct = 0
        incorrect = 0
        for task in task_data:
            if task.answer == correct_answers[task.input_image]:
                correct += 1
            else:
                incorrect += 1
        stat_per_annotator[user_id] = {
            "correct": correct,
            "incorrect": incorrect,
            "score": correct / (correct + incorrect) * 100
        }
    return stat_per_annotator


def load_json_file(path_to_dataset):
    with open(path_to_dataset, 'r') as json_file:
        return json.load(json_file)


def visualize_annotation_durations(users_times, user_to_vendor_id_dict):
    # rearranged the times by the maximum duration, but it can be ordered by any of them
    sorted_users = OrderedDict(sorted(users_times.items(), key=lambda item: item[1]['max']))
    users = [user_to_vendor_id_dict[user] for user in sorted_users.keys()]
    mins = [times['min'] for times in sorted_users.values()]
    maxs = [times['max'] for times in sorted_users.values()]
    avgs = [times['avg'] for times in sorted_users.values()]
    plt.plot(
        users, mins, 'bs',
        users, maxs, 'rs',
        users, avgs, 'gs'
    )
    plt.xlabel('Annotator', color='#34495e')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Durations (ms)', color='#34495e')
    plt.title('Average, min and max annotation times', color='#34495e')
    min_legend = mpatches.Patch(color='red', label='max')
    avg_legend = mpatches.Patch(color='green', label='average')
    max_legend = mpatches.Patch(color='blue', label='min')
    plt.legend(handles=[min_legend, max_legend, avg_legend])
    plt.show()


def visualize_annotated_results_count(work_per_user, user_to_vendor_id_dict):
    sorted_users = OrderedDict(sorted(work_per_user.items(), key=lambda item: item[1]))
    annotators = [user_to_vendor_id_dict[user] for user in sorted_users.keys()]
    plt.bar(annotators, sorted_users.values())
    plt.title('Finished task per annotator')
    plt.xlabel('Annotator')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of tasks')
    plt.show()


def visualize_dataset_balance(true_num, false_num):
    y = [true_num, false_num]
    labels = ["True", "False"]
    plt.pie(y, labels=labels, startangle=90)
    plt.show()


def show_stacked_bar_chart(data, field1, field2, xlabel, ylabel, title, id_to_label_map):
    field1_data = []
    field2_data = []
    if len(id_to_label_map) == 0:
        xticks = list(data.keys())
    else:
        xticks = [id_to_label_map[user] for user in data.keys()]
    for key, value in data.items():
        field1_data.append(value[field1])
        field2_data.append(value[field2])

    plt.bar(xticks, field2_data, color='red', edgecolor='white')
    plt.bar(xticks, field1_data, bottom=field2_data, color='royalblue', edgecolor='white')

    plt.xticks(xticks, rotation=45, ha='right')
    plt.xlabel(xlabel, color='#34495e')
    plt.ylabel(ylabel, color='#34495e')

    plt.title(title, color='#34495e')
    legend1 = mpatches.Patch(color='red', label=f'{field2} task')
    legend2 = mpatches.Patch(color='royalblue', label=f'{field1} tasks')
    plt.legend(handles=[legend1, legend2])

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze dataset.')
    parser.add_argument('--data_json', dest='data_json_path', action='store', required=True,
                        help='path to the json file containing the data.')
    parser.add_argument('--reference_json', dest='reference_json_path', action='store', required=True,
                        help='path to the json file containing the reference data.')
    parser.add_argument('--show_plots', dest='show_plots', action='store_true', default=False,
                        help='show the plots created from the data (default: False)')
    return parser.parse_args()


def main(args):
    # preprocess: load in data, change it to a format more easily usable
    reference_raw = load_json_file(args.reference_json_path)
    annotator_raw = load_json_file(args.data_json_path)

    tasks_per_annotators, id_to_vendor_id = restructure_annotator_data(annotator_raw)

    # Task 1. a)
    print(f"1.a) {len(tasks_per_annotators)} annotator contributed to the dataset.")
    # Task 1. b)
    if args.show_plots:
        durations_per_annotator = get_annotation_durations_statistics(tasks_per_annotators)
        visualize_annotation_durations(durations_per_annotator, id_to_vendor_id)
    # Task 1. c)
    if args.show_plots:
        task_count_per_user = get_annotated_results_counts(tasks_per_annotators)
        visualize_annotated_results_count(task_count_per_user, id_to_vendor_id)
    # Task 1. d)
    answers_per_image = get_annotator_answers_per_image(tasks_per_annotators)
    difficult_images = get_difficult_images(answers_per_image)
    print(f"1.d) the annotators opinion seemed to highly disagree on the results of these images: {difficult_images}.")

    # Task 2.
    statistic_per_annotator, statistic_per_project, corrupt_count, cant_solve_count = \
        get_corruption_cannot_solve_statistic(tasks_per_annotators)
    print(
        f"2.) 'cant_solve' occurs {cant_solve_count} times while 'corrupt_data' occurs {corrupt_count} times in the whole dataset.")
    if args.show_plots:
        show_stacked_bar_chart(statistic_per_annotator, "cant_solve", "corrupt_data", "Annotators",
                               'Unique cases', 'Annotators usage of unique cases', id_to_vendor_id)
        show_stacked_bar_chart(statistic_per_project, "cant_solve", "corrupt_data", "Projects",
                               'Unique cases', 'Unique cases per project', {})

    # Task 3.
    correct_answers = convert_to_correct_answers_dict(reference_raw)
    yes_num = list(correct_answers.values()).count("yes")
    no_num = list(correct_answers.values()).count("no")
    print(f"3.) The reference set seems balanced with {yes_num} positive and {no_num} negative examples.")
    if args.show_plots:
        visualize_dataset_balance(yes_num, no_num)

    # Task 4.
    scores = calculate_annotator_scores(tasks_per_annotators, correct_answers)
    sorted_scores = OrderedDict(
        (user_id, score)
        for user_id, score in sorted(scores.items(), key=lambda item: item[1]["score"])
    )
    print(f"4.) The annotators ordered by the ratio of images they classified correctly")
    for user_id, statistics in sorted_scores.items():
        print(f"    {user_id}: {statistics['score']}")
    if args.show_plots:
        show_stacked_bar_chart(sorted_scores, "correct", "incorrect", "Annotators", 'Finished tasks',
                               'Annotator performance (ordered by correctly classified from all)', id_to_vendor_id)


if __name__ == "__main__":
    main(parse_args())
