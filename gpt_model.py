from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
import base64


def load_png_as_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode("utf-8")

img_path = "/Users/dog/Downloads/32b7ee34cdf66c0b16152b0e3bd3e54c_Normal.png"
img_base64 = load_png_as_base64(img_path)

print(llm.invoke(input=[
    {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}",
                    },
                },
                {
                    "type": "text",
                    "text": """You are a radiologist, you are given an image of a chest x-ray, you need to describe the image in detail, what is the issue that you can find in the image chest? Your response should be in the following format:
                    View: <what view of the chest is this?>,
                    Clinical Impression: <what is the clinical impression for the image?>,
                    Findings: <what is the issue that you can find in the image chest?>,
                    Suggestions: <what are the suggestions for the patient?>
                    """,
                },
            ],
        },
]))

