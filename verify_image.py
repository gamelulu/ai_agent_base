from image_manager.manager import ImageManager
from image_manager.schemas import (
    GoogleImagenKwargs,
    StableDiffusionKwargs,
    MidjourneyKwargs,
    OpenAIDalle3Kwargs
)

def main():
    print("--- 1. Testing Google Image Validation ---")
    try:
        url = ImageManager.create_image(
            prompt="A cinematic mountain lake at sunrise",
            provider="google",
            level=3,
            options=GoogleImagenKwargs(
                number_of_images=2,
                image_size="2K",
                aspect_ratio="16:9",
                add_watermark=False,
                seed=42
            )
        )
        print("Success:", url)
    except Exception as e:
        print("Error:", e)
        
    print("\n--- 2. Testing Stability Image Validation ---")
    try:
        url2 = ImageManager.create_image(
            prompt="Futuristic city",
            provider="stability",
            level=3,
            options=StableDiffusionKwargs(
                negative_prompt="blurry",
                guidance_scale=8.5,
                num_inference_steps=25
            )
        )
        print("Success:", url2)
    except Exception as e:
        print("Error:", e)

    print("\n--- 3. Testing ThirdParty (Midjourney) Validation ---")
    try:
        url3 = ImageManager.create_image(
            prompt="A cat",
            provider="midjourney",
            level=5,
            options=MidjourneyKwargs(
                aspect_ratio="16:9",
                quality="2",
                stylize=500
            )
        )
        print("Success:", url3)
    except Exception as e:
        print("Error:", e)

    print("\n--- 4. Testing OpenAI (DALL-E 3) Validation ---")
    try:
        url4 = ImageManager.create_image(
            prompt="Dog driving a car",
            provider="openai",
            model="dall-e-3",
            options=OpenAIDalle3Kwargs(
                style="vivid",
                quality="hd"
            )
        )
        print("Success:", url4)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
