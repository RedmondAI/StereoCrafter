import os
import sys
import time
from datetime import datetime

def log_time(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)

log_time("Starting imports...")

try:
    # Import torch and initialize CUDA first
    log_time("Importing torch and initializing CUDA...")
    import torch
    cuda_available = torch.cuda.is_available()
    log_time(f"Torch imported successfully. Version: {torch.__version__}")
    log_time(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        try:
            device_name = torch.cuda.get_device_name(0)
            log_time(f"GPU: {device_name}")
            torch.cuda.init()
            log_time("CUDA initialized successfully")
        except Exception as e:
            log_time(f"Warning: CUDA initialization error: {str(e)}")

    log_time("Importing numpy...")
    import numpy as np
    log_time("Numpy imported successfully")

    log_time("Importing Fire...")
    from fire import Fire
    log_time("Fire imported successfully")

    log_time("Importing decord...")
    from decord import VideoReader, cpu
    log_time("Decord imported successfully")

    log_time("Importing torchvision...")
    from torchvision.io import write_video
    log_time("Torchvision imported successfully")

    log_time("Importing transformers...")
    from transformers import CLIPVisionModelWithProjection
    log_time("Transformers imported successfully")

    log_time("Importing diffusers...")
    from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
    log_time("Diffusers imported successfully")

    log_time("Importing custom pipeline...")
    from pipelines.stereo_video_inpainting import StableVideoDiffusionInpaintingPipeline, tensor2vid
    log_time("Custom pipeline imported successfully")

except Exception as e:
    log_time(f"Error during imports: {str(e)}")
    log_time(f"Error type: {type(e)}")
    log_time(f"Error location: {sys.exc_info()[-1].tb_lineno}")
    raise

log_time("All imports successful")

def blend_h(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    weight_b = (torch.arange(overlap_size).view(1, 1, 1, -1) / overlap_size).to(
        b.device
    )
    b[:, :, :, :overlap_size] = (1 - weight_b) * a[
        :, :, :, -overlap_size:
    ] + weight_b * b[:, :, :, :overlap_size]
    return b

def blend_v(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    weight_b = (torch.arange(overlap_size).view(1, 1, -1, 1) / overlap_size).to(
        b.device
    )
    b[:, :, :overlap_size, :] = (1 - weight_b) * a[
        :, :, -overlap_size:, :
    ] + weight_b * b[:, :, :overlap_size, :]
    return b

def spatial_tiled_process(
    cond_frames,
    mask_frames,
    process_func,
    tile_num,
    spatial_n_compress=4,
    debug=False,
    **kargs,
):
    height = cond_frames.shape[2]
    width = cond_frames.shape[3]
    if debug:
        print(f"\nInput dimensions: height={height}, width={width}")

    # Base overlap in pixel space - must be divisible by 8 for VAE
    base_overlap = 128
    if debug:
        print(f"Base overlap: {base_overlap}")
    
    # Calculate minimum tile size needed
    min_tile_size = (
        height // tile_num + base_overlap * (tile_num - 1) // tile_num,
        width // tile_num + base_overlap * (tile_num - 1) // tile_num
    )
    if debug:
        print(f"Minimum tile size: {min_tile_size}")
    
    # Round up to nearest multiple of 128 (VAE requires multiples of 8)
    tile_size = (
        ((min_tile_size[0] + 127) // 128) * 128,
        ((min_tile_size[1] + 127) // 128) * 128
    )
    if debug:
        print(f"Rounded tile size: {tile_size}")
    
    # Overlap in pixel space
    tile_overlap = (base_overlap, base_overlap)
    if debug:
        print(f"Tile overlap: {tile_overlap}")
    
    # Calculate stride
    tile_stride = (
        tile_size[0] - tile_overlap[0],
        tile_size[1] - tile_overlap[1]
    )
    if debug:
        print(f"Tile stride: {tile_stride}")
    
    # Adjust input dimensions to match tile layout
    total_height = tile_stride[0] * tile_num + tile_overlap[0]
    total_width = tile_stride[1] * tile_num + tile_overlap[1]
    if debug:
        print(f"Total dimensions needed: height={total_height}, width={total_width}")
    
    if total_height > height or total_width > width:
        # Pad input if needed
        pad_height = max(0, total_height - height)
        pad_width = max(0, total_width - width)
        if debug:
            print(f"Padding needed: height={pad_height}, width={pad_width}")
        cond_frames = torch.nn.functional.pad(cond_frames, (0, pad_width, 0, pad_height))
        mask_frames = torch.nn.functional.pad(mask_frames, (0, pad_width, 0, pad_height))
    else:
        # Crop input if needed
        if debug:
            print(f"Cropping to: height={total_height}, width={total_width}")
        cond_frames = cond_frames[:, :, :total_height, :total_width]
        mask_frames = mask_frames[:, :, :total_height, :total_width]
    
    if debug:
        print(f"Adjusted input dimensions: {cond_frames.shape}")
    
    cols = []
    for i in range(tile_num):
        rows = []
        for j in range(tile_num):
            start_h = i * tile_stride[0]
            start_w = j * tile_stride[1]
            
            cond_tile = cond_frames[
                :, :,
                start_h:start_h + tile_size[0],
                start_w:start_w + tile_size[1]
            ]
            mask_tile = mask_frames[
                :, :,
                start_h:start_h + tile_size[0],
                start_w:start_w + tile_size[1]
            ]
            
            if debug:
                print(f"\nTile ({i},{j}) dimensions:")
                print(f"  Start position: ({start_h}, {start_w})")
                print(f"  Tile shape: {cond_tile.shape}")

            tile = process_func(
                frames=cond_tile,
                frames_mask=mask_tile,
                height=cond_tile.shape[2],
                width=cond_tile.shape[3],
                num_frames=len(cond_tile),
                output_type="latent",
                **kargs,
            ).frames[0]
            
            if debug:
                print(f"  Latent tile shape: {tile.shape}")
            rows.append(tile)
        cols.append(rows)

    # VAE uses 8x compression for both dimensions
    vae_scale = 8
    
    # Calculate latent space dimensions
    latent_stride = (
        tile_stride[0] // vae_scale,
        tile_stride[1] // vae_scale
    )
    latent_overlap = (
        tile_overlap[0] // vae_scale,
        tile_overlap[1] // vae_scale
    )
    if debug:
        print(f"\nLatent space dimensions:")
        print(f"  Stride: {latent_stride}")
        print(f"  Overlap: {latent_overlap}")

    # Blend tiles in latent space
    results_cols = []
    for i, rows in enumerate(cols):
        results_rows = []
        for j, tile in enumerate(rows):
            if i > 0:
                if debug:
                    print(f"\nVertical blend at ({i},{j}):")
                    print(f"  Previous tile shape: {cols[i-1][j].shape}")
                    print(f"  Current tile shape: {tile.shape}")
                    print(f"  Overlap size: {latent_overlap[0]}")
                tile = blend_v(cols[i - 1][j], tile, latent_overlap[0])
            if j > 0:
                if debug:
                    print(f"\nHorizontal blend at ({i},{j}):")
                    print(f"  Previous tile shape: {rows[j-1].shape}")
                    print(f"  Current tile shape: {tile.shape}")
                    print(f"  Overlap size: {latent_overlap[1]}")
                tile = blend_h(rows[j - 1], tile, latent_overlap[1])
            results_rows.append(tile)
        results_cols.append(results_rows)

    # Combine tiles
    pixels = []
    for i, rows in enumerate(results_cols):
        for j, tile in enumerate(rows):
            if i < len(results_cols) - 1:
                tile = tile[:, :, :latent_stride[0], :]
            if j < len(rows) - 1:
                tile = tile[:, :, :, :latent_stride[1]]
            if debug:
                print(f"\nFinal tile ({i},{j}) shape: {tile.shape}")
            rows[j] = tile
        pixels.append(torch.cat(rows, dim=3))
    x = torch.cat(pixels, dim=2)
    if debug:
        print(f"\nFinal output shape: {x.shape}")
    
    return x

def main(
    pre_trained_path,
    unet_path,
    input_video_path,
    save_dir,
    frames_chunk=23,
    overlap=8,
    tile_num=1,
    spatial_n_compress=8,
    num_inference_steps=50,
    debug=False
):
    log_time("Starting script")
    log_time(f"Loading models from {pre_trained_path} and {unet_path}")
    
    log_time("Loading CLIP image encoder...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pre_trained_path,
        subfolder="image_encoder",
        variant="fp16",
        torch_dtype=torch.float16
    )
    log_time("CLIP image encoder loaded")

    log_time("Loading VAE...")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pre_trained_path, 
        subfolder="vae", 
        variant="fp16", 
        torch_dtype=torch.float16
    )
    log_time("VAE loaded")

    log_time("Loading UNet...")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        unet_path,
        subfolder="unet_diffusers",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    log_time("UNet loaded")

    log_time("Setting up models...")
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    log_time("Creating pipeline...")
    pipeline = StableVideoDiffusionInpaintingPipeline.from_pretrained(
        pre_trained_path,
        image_encoder=image_encoder,
        vae=vae,
        unet=unet,
        torch_dtype=torch.float16,
    )
    pipeline = pipeline.to("cuda")
    log_time("Pipeline created and moved to GPU")

    os.makedirs(save_dir, exist_ok=True)
    video_name = input_video_path.split("/")[-1].replace(".mp4", "").replace("_splatting_results", "") + "_inpainting_results"

    log_time("Loading video...")
    video_reader = VideoReader(input_video_path, ctx=cpu(0))
    fps = video_reader.get_avg_fps()
    log_time(f"Video loaded: {len(video_reader)} frames at {fps} FPS")
    frame_indices = list(range(len(video_reader)))
    frames = video_reader.get_batch(frame_indices)
    num_frames = len(video_reader)

    # [t,h,w,c] -> [t,c,h,w]
    frames = (
        torch.tensor(frames.asnumpy()).permute(0, 3, 1, 2).float()
    )  

    height, width = frames.shape[2] // 2, frames.shape[3] // 2
    frames_left = frames[:, :, :height, :width]
    frames_mask = frames[:, :, height:, :width]
    frames_warpped = frames[:, :, height:, width:]

    # ---------------------------------------------------------
    # REPLACED the triple nested loop with a single pass over W
    # to fill black pixels (<10) from the pixel on the left.
    # ---------------------------------------------------------
    threshold = 10.0  # Compare in 0..255 range
    T = frames_warpped.shape[0]  # number of frames
    C = frames_warpped.shape[1]  # channels (likely 3)
    H = frames_warpped.shape[2]
    W = frames_warpped.shape[3]

    # Go from left to right
    for w_ in range(1, W):
        # Find which pixels in column w_ are black in all channels
        # shape: [T, H, C], so we do .all(dim=1) across channels -> [T, H]
        black_mask = (frames_warpped[:, :, :, w_] < threshold).all(dim=1)
        # Fill them from column w_ - 1
        for c_ in range(C):
            frames_warpped[:, c_, :, w_][black_mask] = frames_warpped[:, c_, :, w_ - 1][black_mask]
    # ---------------------------------------------------------

    frames = torch.cat([frames_warpped, frames_left, frames_mask], dim=0)

    # Dimensions will be adjusted in spatial_tiled_process
    frames = frames / 255.0
    frames_warpped, frames_left, frames_mask = torch.chunk(frames, chunks=3, dim=0)
    frames_mask = frames_mask.mean(dim=1, keepdim=True)

    results = []
    generated = None
    start_time = time.time()  # Start timing
    for i in range(0, num_frames, frames_chunk - overlap):

        if i + overlap >= frames_warpped.shape[0]:
            break

        if generated is not None and i + frames_chunk > frames_warpped.shape[0]:
            cur_i = max(frames_warpped.shape[0] + overlap - frames_chunk, 0)
            cur_overlap = i - cur_i + overlap
        else:
            cur_i = i
            cur_overlap = overlap

        input_frames_i = frames_warpped[cur_i : cur_i + frames_chunk].clone()
        mask_frames_i = frames_mask[cur_i : cur_i + frames_chunk]

        if generated is not None:
            try:
                input_frames_i[:cur_overlap] = generated[-cur_overlap:]
            except Exception as e:
                print(e)
                print(
                    f"i: {i}, cur_i: {cur_i}, cur_overlap: {cur_overlap}, input_frames_i: {input_frames_i.shape}, generated: {generated.shape}"
                )

        video_latents = spatial_tiled_process(
            input_frames_i,
            mask_frames_i,
            pipeline,
            tile_num,
            spatial_n_compress=spatial_n_compress,
            debug=debug,
            min_guidance_scale=1.01,
            max_guidance_scale=1.01,
            decode_chunk_size=8,
            fps=7,
            motion_bucket_id=127,
            noise_aug_strength=0.0,
            num_inference_steps=num_inference_steps,
        )

        video_latents = video_latents.unsqueeze(0)
        if video_latents == torch.float16:
            pipeline.vae.to(dtype=torch.float16)

        video_frames = pipeline.decode_latents(video_latents, num_frames=video_latents.shape[1], decode_chunk_size=2)
        video_frames = tensor2vid(video_frames, pipeline.image_processor, output_type="pil")[0]

        for j in range(len(video_frames)):
            img = video_frames[j]
            video_frames[j] = (
                torch.tensor(np.array(img)).permute(2, 0, 1).to(dtype=torch.float32)
                / 255.0
            )
        generated = torch.stack(video_frames)
        if i != 0:
            generated = generated[cur_overlap:]
        results.append(generated)

    frames_output = torch.cat(results, dim=0).cpu()
    processing_time = round(time.time() - start_time, 2)  # End timing
    log_time(f"Frame processing took {processing_time} seconds")

    '''
    video_mask = frames_mask.repeat(1, 3, 1, 1)
    top = torch.cat([frames_left, frames_warpped], dim=3)
    bottom = torch.cat([video_mask, frames_output], dim=3)

    frames_all = torch.cat([top, bottom], dim=2)
    frames_all_path = os.path.join(save_dir, f"{video_name}.mp4")
    os.makedirs(os.path.dirname(frames_all_path), exist_ok=True)

    frames_all = (frames_all * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu()
    write_video(
        frames_all_path,
        frames_all,
        fps=fps,
        video_codec="h264",
        options={"crf": "16"},
    )
    '''

    #frames_sbs = torch.cat([frames_left, frames_output], dim=3)
    #frames_sbs_path = os.path.join(save_dir, f"{video_name}_sbs.mp4")
    #frames_sbs = torch.cat([frames_left, frames_output], dim=3)
    frames_sbs_path = os.path.join(save_dir, f"{video_name}_inpainted_{processing_time}s.mp4")
    frames_sbs = ((frames_output * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu())
    write_video(
        frames_sbs_path,
        frames_sbs,
        fps=fps,
        video_codec="h264",
        options={"crf": "5"},
    )

    # vid_left = (frames_left * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu().numpy()
    # vid_right = (frames_output * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu().numpy()

    # vid_left[:, :, :, 1] = 0
    # vid_left[:, :, :, 2] = 0
    # vid_right[:, :, :, 0] = 0

    # vid_anaglyph = vid_left + vid_right
    # vid_anaglyph_path = os.path.join(save_dir, f"{video_name}_anaglyph.mp4")
    # write_video(
    #     vid_anaglyph_path,
    #     vid_anaglyph,
    #     fps=fps,
    #     video_codec="h264",
    #     options={"crf": "10"},
    # )

if __name__ == "__main__":
    Fire(main)