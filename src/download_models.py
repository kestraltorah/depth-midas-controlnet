from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter
from controlnet_aux.midas import MidasDetector

def download_models():
    print("Downloading models...")
    
    print("1. Downloading T2I Adapter...")
    adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-depth-midas-sdxl-1.0",
        cache_dir="/workspace/models/adapter"
    )
    
    print("2. Downloading SDXL base model...")
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        adapter=adapter,
        cache_dir="/workspace/models/sdxl"
    )
    
    print("3. Downloading Midas model...")
    midas = MidasDetector.from_pretrained(
        "valhalla/t2iadapter-aux-models",
        filename="dpt_large_384.pt",
        cache_dir="/workspace/models/midas"
    )
    
    print("All models downloaded successfully!")

if __name__ == "__main__":
    download_models()