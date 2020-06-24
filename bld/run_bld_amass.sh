blender /media/F/thesis/motion_capture/bld/amass_syn_hdri_maker.blend \
--background \
--log-level 3 \
--python ./syn_motion_videos.py -- \
--smplx /media/F/projects/moveai/codes/run_data/amass/smpl/models_smplx_v1_0/models/smplx \
--amass /media/F/projects/moveai/codes/run_data/amass/motion_data/ \
--texture_dir /media/F/projects/moveai/codes/run_data/amass/textures \
--out_dir /media/F/projects/moveai/codes/run_data/amass/syn/ \
--n_cams 5 \
--gen_data \
--gen_video \
--max_anim_frame 1000 \
--amass_csv /media/F/projects/moveai/codes/run_data/amass/anims.csv