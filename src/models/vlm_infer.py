import re

def classify_crash_images(model, processor, before_image, impact_image):
    system_prompt = (
        "Question: Does the image show the moment of impact? "
        "What type of vehicle collision is best described between the two images?\n"
        "Options:\n 1) 'head-on'\n 2) 'rear-end'\n 3) 'sideswipe'\n"
        " 4) 't-bone'\n 5) 'single'\n 6) 'no accident'.\nOutput only the digit."
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "text": "Image shows vehicles approaching"},
            {"type": "image", "text": "Does the image show the moment of impact?"},
            {"type": "text",  "text": system_prompt},
        ]
    }]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[before_image, impact_image], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    generated_ids = model.generate(**inputs, max_new_tokens=10)
    trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], generated_ids)]
    output_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip().lower()

    class_mapping = {
        "1": "head-on", "2": "rear-end", "3": "sideswipe",
        "4": "t-bone",  "5": "single",   "6": "no accident",
    }
    match = re.search(r"[1-6]", output_text)
    predicted_type = class_mapping[match.group(0)] if match else "unknown"
    if predicted_type in ("unknown", "no accident"):
        predicted_type = "single"
    return predicted_type