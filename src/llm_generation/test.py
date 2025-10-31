# import google.generativeai as genai
# import os
# from dotenv import load_dotenv

# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# print("\n✅ Available models:")
# for m in genai.list_models():
#     if "generateContent" in m.supported_generation_methods:
#         print("-", m.name)

import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("models/gemini-2.5-flash")

response = model.generate_content("Write a short outreach email for a nonprofit organization seeking donors.")
print(response.text)
