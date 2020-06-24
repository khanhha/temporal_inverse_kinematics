import requests
from tqdm import tqdm
from pathlib import Path
import csv


def download_subject_txts():
    outdir = './data'
    for sub_id in tqdm(range(1, 145)):
        txt_url = f'http://mocap.cs.cmu.edu/search.php?subjectnumber={sub_id}' + '&motion=%%%&maincat=%&subcat=%&subtext=yes'
        r = requests.get(txt_url, allow_redirects=True)
        rows = r.content.decode("utf-8")
        with open(f'{outdir}/{sub_id}.txt', 'wt') as file:
            file.write(rows)


def is_wanted_subject(anno: str):
    anno = anno.lower()
    trial_keywords = ["climb", "swing", "dance", "basketball", "soccer", "tai chi", "jump", "recreation",
                      "football", "salsa", "golf", "kick", "hopscotch", "cartwheels", "acrobatics", "sport", "bending",
                      "swimming", "rolling", "Jackson", "action walk", "flip", "breakdance"]
    for kw in trial_keywords:
        if kw.lower() in anno:
            return True
    return False


def is_wanted_trial(anno: str):
    anno = anno.lower()
    trial_keywords = ["jump", "dance", 'art', "martial", "acrobatics", "bending", "sport", "punch", "jog",
                      "swing", "tai chi", "basketball", "cartwheel", "soccer", "climb", "stretch", "gymnastics",
                      "leap", "spin", "hang", "twirl", "hop", "salsa", "swing", "ball", "kick", "lifting", "run",
                      "hopscotch"]
    for kw in trial_keywords:
        if kw in anno:
            return True
    return False


def run_main():
    subjects = []
    data_dir = f'./data/cmu_labels'
    for path in Path(data_dir).glob('*.txt'):
        with open(str(path), 'rt') as file:
            rows = file.readlines()

        rows = [r.strip() for r in rows]
        sub_id = int(path.stem)
        sub_lbl = [r for r in rows if 'subject' in r.lower() and "#" in r.lower()]
        if len(sub_lbl) != 1:
            print(f'bad file: ', path)
            continue

        rows = [r for r in rows if len(r.split('\t')) >= 5]
        trials = []
        for r in rows:
            parts = r.split('\t')
            assert len(parts[0]) and len(parts[-1])
            trials.append((parts[0], parts[-1]))

        subjects.append((sub_id, sub_lbl[0], trials))

    subjects = [sub for sub in subjects if is_wanted_subject(sub[1])]

    all_trials = []
    for sub in subjects:
        all_trials.extend(sub[2])
    print(f'total trials = {len(all_trials)}')
    with open('data/cmu_action_list.csv', 'w') as file:
        writer = csv.writer(file, delimiter='|')
        writer.writerows(all_trials)

    generate_cmu_csv_action_lists([t[0] for t in all_trials], batch_size=3)


def generate_cmu_csv_action_lists(trial_names, batch_size=10):
    n_trials = len(trial_names)
    n_batches = n_trials // batch_size
    out_dir = '/media/F/projects/moveai/codes/run_data/amass/csv_batches'
    cmu_data_dir = '/media/F/projects/moveai/codes/run_data/amass/motion_data/CMU/CMU'
    all_files = {path.stem: path for path in Path(cmu_data_dir).rglob('*.npz')}
    for i in range(n_batches):
        batch = trial_names[i * batch_size: (i + 1) * batch_size]
        if batch:
            batch = [f'{t_name}_poses' for t_name in batch]
            batch_1 = []
            for name in batch:
                if name not in all_files:
                    print(f'animation {name} does not exist')
                else:
                    batch_1.append(name)

            if batch_1:
                with open(f'{out_dir}/batch_{i}.csv', 'w') as file:
                    writer = csv.writer(file)
                    writer.writerows(zip(batch_1))


if __name__ == "__main__":
    run_main()
