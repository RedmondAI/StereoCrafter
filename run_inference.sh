python depth_splatting_inference.py \
   --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1\
   --unet_path ./weights/DepthCrafter \
   --input_video_path ./source_video/12a.mp4 \
   --output_video_path ./outputs/12a_splatting_results.mp4


python -v inpainting_inference.py \
    --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
    --unet_path ./weights/StereoCrafter \
    --input_video_path ./outputs/12a_splatting_results.mp4 \
    --save_dir ./outputs