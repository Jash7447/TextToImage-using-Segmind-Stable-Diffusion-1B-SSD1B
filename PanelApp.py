import panel as pn
import torch
from diffusers import StableDiffusionXLPipeline

# Initialize the model
pipe = StableDiffusionXLPipeline.from_pretrained(
    "segmind/SSD-1B",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
pipe.to("cuda")

# Function to generate the image
def generate_image(prompt, neg_prompt):
    image = pipe(prompt=prompt, negative_prompt=neg_prompt).images[0]
    return image

# Create a Panel app
pn.extension()

prompt_input = pn.widgets.TextInput(placeholder="Enter your prompt")
neg_prompt_input = pn.widgets.TextInput(placeholder="Enter your negative prompt")

image_viewer = pn.pane.Image()

def update_image(event):
    prompt = prompt_input.value
    neg_prompt = neg_prompt_input.value
    image = generate_image(prompt, neg_prompt)
    image_viewer.object = image

generate_button = pn.widgets.Button(name="Generate Image", button_type="primary")
generate_button.on_click(update_image)

app = pn.Column(
    prompt_input,
    neg_prompt_input,
    generate_button,
    image_viewer,
)

# Run the Panel app
app.servable()
