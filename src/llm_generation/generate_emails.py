# src/llm_generation/generate_emails_gemini.py
"""
Generate outreach email templates for each cluster
using Google's Gemini 2.5 API.
"""

import os
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from tqdm import tqdm  

# === 1. Load environment variables ===
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

genai.configure(api_key=api_key)

# === 2. Load clustered dataset ===
input_path = "data/processed/clustered_contacts.csv"
df = pd.read_csv(input_path)
print(f"Loaded {len(df)} records from {input_path}")

# === 3. Define Gemini model ===
model = genai.GenerativeModel("models/gemini-2.5-flash")

# === 4. Email generation function ===
def generate_email(cluster_id, company_examples, tone):
    prompt = f"""
    You are assisting an NGO called Second Life e.V.
    Please write a {tone} outreach email template targeting donor companies.

    Cluster ID: {cluster_id}
    Example companies: {', '.join(company_examples) if company_examples else 'N/A'}

    - Keep it short (120–150 words)
    - Clear, persuasive, and mission-aligned
    - Adjust tone to suit the cluster profile (e.g., low/medium/high potential)
    - Include a simple call-to-action
    """

    try:
        response = model.generate_content(prompt)
        return response.text.strip() if response and response.text else "(Empty response)"
    except Exception as e:
        print(f"⚠️ Error generating email for cluster {cluster_id}, tone={tone}: {e}")
        return "(Generation failed)"

# === 5. Generate templates for each cluster and tone ===
emails = []
unique_clusters = df["cluster"].unique()

for cluster_id in tqdm(unique_clusters, desc="Generating emails by cluster"):
    examples = df[df["cluster"] == cluster_id]["name"].dropna().head(3).tolist()
    for tone in ["formal", "conversational", "storytelling"]:
        text = generate_email(cluster_id, examples, tone)
        emails.append({
            "cluster": cluster_id,
            "tone": tone,
            "example_companies": ", ".join(examples),
            "template": text
        })

# === 6. Save results ===
output_path = "data/processed/email_templates.csv"
email_df = pd.DataFrame(emails)
email_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\nEmail templates generated and saved to: {output_path}")
print(f"Total templates created: {len(email_df)}")
