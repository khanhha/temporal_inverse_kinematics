from pathlib import Path
import fire
import subprocess
import os


def load_log(fpath):
    if Path(fpath).exists():
        with open(fpath, 'r') as file:
            return [line.replace('\n', '') for line in file.readlines()]
    else:
        return []


def save_log(fpath, logs):
    with open(fpath, 'w+') as file:
        for log in logs:
            file.write(f'{log}\n')


def run_main(csv_dir, log_dir, n_cam: int, batch_size: int, max_anim_frame: int):
    csv_paths = [path for path in Path(csv_dir).glob('*.csv')]
    n_batches = len(csv_paths) // batch_size
    flog_proc_path = f'{log_dir}/processed_files.txt'
    processed_files = load_log(flog_proc_path)

    os.makedirs(log_dir, exist_ok=True)

    for bi in range(n_batches):
        files = csv_paths[bi * batch_size:(bi + 1) * batch_size]
        if not files:
            break

        files = [f for f in files if f not in processed_files]

        pipes = []
        for file in files:
            log_path = f'{log_dir}/{file.stem}.log'

            p = subprocess.Popen(['blender', "/media/F/thesis/motion_capture/bld/amass_syn_hdri_maker.blend",
                                  "--background",
                                  "--log-level", "3",
                                  "--python", "/media/F/thesis/motion_capture/bld/syn_motion_videos.py",
                                  "--",
                                  "--smplx",
                                  "/media/F/projects/moveai/codes/run_data/amass/smpl/models_smplx_v1_0/models/smplx",
                                  "--amass", "/media/F/projects/moveai/codes/run_data/amass/motion_data/",
                                  "--texture_dir", "/media/F/projects/moveai/codes/run_data/amass/textures",
                                  "--out_dir", "/media/F/projects/moveai/codes/run_data/amass/syn/",
                                  "--n_cams", f"{n_cam}",
                                  "--gen_data",
                                  "--gen_video",
                                  "--max_anim_frame", f"{max_anim_frame}",
                                  "--amass_csv", f"{file}",
                                  "--log", log_path])
            pipes.append(p)

        for p, file in zip(pipes, files):
            p.wait()
            processed_files.append(file.name)
            save_log(flog_proc_path, processed_files)


if __name__ == "__main__":
    fire.Fire(run_main)
