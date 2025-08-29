from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)
CORS(app)

# Load expert data
def load_experts():
    try:
        df = pd.read_csv('/Users/markusenenkel/Desktop/CST_App/fake_climate_researchers.csv')
        experts = []
        
        for idx, row in df.iterrows():
            # Combine expertise fields into tags
            expertise_tags = []
            for col in ['Expertise A', 'Expertise B', 'Expertise C']:
                if pd.notna(row[col]) and row[col].strip():
                    expertise_tags.append(row[col].strip())
            
            expert = {
                'id': f"e{idx:02d}",
                'name': row['Name'],
                'affiliation': 'academia',  # Default since not in data
                'expertise_tags': ','.join(expertise_tags),
                'bio': f"Expert in {', '.join(expertise_tags)}",
                'contact_email': row['Email'],
                'location': 'Unknown',
                'seniority': 'mid-level'  # Default
            }
            experts.append(expert)
        
        return experts
    except Exception as e:
        print(f"Error loading experts: {e}")
        return []

# Initialize experts
experts_data = load_experts()

def calculate_semantic_similarity(question, thematic_area, expert_tags, expert_bio):
    """Calculate semantic similarity between question and expert expertise"""
    # Combine question and thematic area
    query_text = f"{question} {thematic_area}".lower()
    
    # Combine expert tags and bio
    expert_text = f"{expert_tags} {expert_bio}".lower()
    
    # Primary hazard terms that must match for high scores
    primary_hazards = [
        'drought', 'flood', 'storm', 'heat wave', 'earthquake', 'mudslide', 'multi-hazard',
        'flooding', 'droughts', 'storms', 'heat waves', 'earthquakes', 'mudslides',
        'drought risk', 'flood forecasting', 'flood risk', 'storm forecasting'
    ]
    
    # Secondary climate terms for broader matching
    climate_terms = [
        'maize', 'agriculture', 'farming', 'crop', 'west africa', 'africa', 'sahel',
        'carbon', 'sequestration', 'emissions', 'greenhouse gas', 'co2',
        'adaptation', 'mitigation', 'resilience', 'vulnerability',
        'anticipatory action', 'early warning', 'early action', 'preparedness',
        'climate', 'weather', 'temperature', 'precipitation', 'rainfall',
        'renewable', 'energy', 'solar', 'wind', 'hydro', 'geothermal',
        'atmospheric', 'air quality', 'pollution', 'aerosol',
        'oceanography', 'ocean', 'marine', 'coastal', 'sea level',
        'hydrology', 'water', 'river', 'lake', 'groundwater',
        'meteorology', 'forecasting', 'prediction', 'modeling',
        'ecology', 'ecosystem', 'biodiversity', 'conservation',
        'environmental', 'sustainability', 'green', 'natural',
        'policy', 'governance', 'regulation', 'planning',
        'modeling', 'simulation', 'computer', 'algorithm',
        'data', 'analysis', 'statistics', 'research',
        'risk', 'assessment', 'management', 'monitoring',
        'communication', 'education', 'outreach', 'public',
        'economics', 'finance', 'investment', 'cost',
        'geophysics', 'geology', 'seismic', 'tectonic',
        'paleoclimatology', 'historical', 'proxy', 'reconstruction',
        'remote', 'sensing', 'satellite', 'gps', 'gis',
        'development', 'sustainable', 'urban', 'rural',
        'health', 'public health', 'epidemiology',
        'transport', 'mobility', 'infrastructure',
        'forest', 'deforestation', 'afforestation',
        'soil', 'erosion', 'degradation', 'fertility',
        
        # Additional terms from CSV expertise areas
        'disaster risk finance', 'drought risk assessment', 'climate risk assessment',
        'climate change adaptation', 'climate communication', 'climate data analysis',
        'climate modeling', 'environmental economics', 'environmental policy',
        'flood forecasting', 'global warming', 'greenhouse gas emissions',
        'machine learning', 'artificial intelligence', 'data science',
        'statistical analysis', 'computational modeling', 'digital tools',
        'technology', 'innovation', 'sustainable development', 'renewable energy',
        'carbon sequestration', 'atmospheric science', 'geophysics'
    ]
    
    # Check for primary hazard matches first
    query_hazards = []
    expert_hazards = []
    
    # Extract hazards from query
    for hazard in primary_hazards:
        if hazard in query_text:
            query_hazards.append(hazard)
    
    # Extract hazards from expert expertise
    for hazard in primary_hazards:
        if hazard in expert_text:
            expert_hazards.append(hazard)
    
    # STRICT HAZARD MATCHING: If query mentions a specific hazard, expert MUST have that exact hazard
    if query_hazards:
        # Check if expert has ANY of the mentioned hazards
        hazard_matches = set(query_hazards).intersection(set(expert_hazards))
        if not hazard_matches:
            # Debug: Print why expert was rejected
            print(f"DEBUG: Expert rejected - Query hazards: {query_hazards}, Expert hazards: {expert_hazards}")
            return 0.0  # No match - expert doesn't have the requested hazard expertise
        
        # If expert has the hazard, give high score but still check other terms
        base_score = 0.8  # High base score for hazard match
        print(f"DEBUG: Expert accepted - Query hazards: {query_hazards}, Expert hazards: {expert_hazards}, Matches: {hazard_matches}")
    else:
        base_score = 0.0  # No hazard mentioned, start with 0
    
    # Extract other key terms from question
    question_terms = set()
    for term in climate_terms:
        if term in query_text:
            question_terms.add(term)
    
    # Extract key terms from expert expertise
    expert_terms = set()
    for term in climate_terms:
        if term in expert_text:
            expert_terms.add(term)
    
    # Check for exact word matches - give 100% score
    query_words = query_text.lower().split()
    expert_words = expert_text.lower().split()
    
    for query_word in query_words:
        query_word_clean = query_word.strip('.,!?;:')
        if len(query_word_clean) > 3:  # Only significant words
            for expert_word in expert_words:
                expert_word_clean = expert_word.strip('.,!?;:')
                if query_word_clean == expert_word_clean:
                    return 1.0  # 100% match for exact word
    
    # If no hazard mentioned, calculate similarity based on other terms
    if not query_hazards:
        if not question_terms or not expert_terms:
            return 0.0
        
        intersection = len(question_terms.intersection(expert_terms))
        union = len(question_terms.union(expert_terms))
        
        if union == 0:
            return 0.0
        
        base_score = intersection / union
    else:
        # Hazard was mentioned and matched, now add bonus for other term matches
        if question_terms and expert_terms:
            intersection = len(question_terms.intersection(expert_terms))
            union = len(question_terms.union(expert_terms))
            
            if union > 0:
                term_bonus = intersection / union * 0.2  # Small bonus for additional term matches
                base_score += term_bonus
    
    # Additional boost for thematic area matches
    if thematic_area.lower() in expert_text:
        base_score *= 1.1
    
    return min(1.0, base_score)

def calculate_confidence(semantic_score, data_completeness):
    """Calculate confidence score (0-100)"""
    semantic_weight = 0.8
    completeness_weight = 0.2
    
    confidence = (semantic_score * semantic_weight + data_completeness * completeness_weight) * 100
    return min(100, max(0, int(confidence)))

def get_expert_recommendations(question, thematic_area, user_affiliation, experts):
    """Get expert recommendations based on question and user context"""
    recommendations = []
    
    for expert in experts:
        # Calculate semantic similarity
        semantic_score = calculate_semantic_similarity(
            question, thematic_area, expert['expertise_tags'], expert['bio']
        )
        
        # Calculate data completeness (simple heuristic)
        completeness = 0.0
        if expert['expertise_tags']:
            completeness += 0.5
        if expert['bio']:
            completeness += 0.3
        if expert['contact_email']:
            completeness += 0.2
        
        # Affiliation bonus (small boost for matching affiliations)
        affiliation_bonus = 0.0
        if user_affiliation and expert['affiliation']:
            if user_affiliation.lower() == expert['affiliation'].lower():
                affiliation_bonus = 0.1
        
        # Calculate match score
        match_score = min(100, int((semantic_score + affiliation_bonus) * 100))
        
        # Only include experts with reasonable match scores
        if match_score > 25:  # Very high threshold for strict matching
            # Generate reason for recommendation
            expertise_list = expert['expertise_tags'].split(',')
            if len(expertise_list) > 1:
                reason = f"Expertise in {expertise_list[0]} and related areas"
            else:
                reason = f"Expertise in {expertise_list[0]}"
            
            recommendation = {
                'id': expert['id'],
                'name': expert['name'],
                'affiliation': expert['affiliation'].title(),
                'match_score': match_score,
                'reason': reason,
                'contact_email': expert['contact_email'],
                'expertise_summary': ', '.join(expert['expertise_tags'].split(',')[:3]),
                'contact_info': f"Contact: {expert['name']} at {expert['contact_email']}"
            }
            recommendations.append(recommendation)
    
    # Sort by match score and return top 3
    recommendations.sort(key=lambda x: x['match_score'], reverse=True)
    return recommendations[:3]

def generate_answer(question, thematic_area, user_affiliation=None):
    """Generate a concise answer using ChatGPT API"""
    try:
        # Create a system prompt for the climate science translator
        system_prompt = f"""You are a Climate Science Translator Assistant. Your role is to provide concise, accurate, and accessible answers to climate science questions.

Context:
- User's affiliation: {user_affiliation or 'general'}
- Thematic area of interest: {thematic_area}

Guidelines:
1. Provide clear, concise answers in plain language
2. Use scientific accuracy while being accessible to non-experts
3. Consider the user's affiliation context when appropriate
4. Keep answers focused and relevant to the specific question
5. Use active voice and clear structure
6. Avoid jargon unless necessary, and explain technical terms when used

Please answer the following question:"""

        # Create the user message
        user_message = f"{question}"

        # Call ChatGPT API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error calling ChatGPT API: {e}")
        # Fallback to a simple answer if API fails
        return f"I apologize, but I'm unable to provide a detailed answer at the moment. Please try again later or contact a climate science expert directly."

def generate_follow_up(question, thematic_area):
    """Generate a follow-up question"""
    return ""

@app.route('/api/assistant', methods=['POST'])
def assistant():
    try:
        data = request.get_json()
        
        # Extract data
        onboarding = data.get('onboarding', {})
        question = data.get('question', '')
        experts = data.get('experts', experts_data)
        
        # Validate required fields
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        # Check if onboarding is complete
        if not onboarding or not onboarding.get('affiliation') or not onboarding.get('thematic_area'):
            return jsonify({
                'error': 'Onboarding required',
                'missing_fields': ['affiliation', 'thematic_area', 'contact']
            }), 400
        
        # Generate answer
        answer = generate_answer(question, onboarding['thematic_area'], onboarding.get('affiliation'))
        
        # Get expert recommendations
        recommendations = get_expert_recommendations(
            question, 
            onboarding['thematic_area'], 
            onboarding['affiliation'], 
            experts
        )
        
        # Calculate overall confidence
        if recommendations:
            avg_match = sum(r['match_score'] for r in recommendations) / len(recommendations)
            confidence = calculate_confidence(avg_match / 100, 0.8)  # Assume good data completeness
        else:
            confidence = 30  # Low confidence if no good matches
        
        # Generate follow-up
        follow_up = generate_follow_up(question, onboarding['thematic_area'])
        
        # Prepare response
        response = {
            'answer': answer,
            'confidence': confidence,
            'recommended_experts': recommendations,
            'follow_up': follow_up,
            'debug': {
                'matched_tags': [r['reason'].split(' in ')[1].split(' and')[0] for r in recommendations if ' in ' in r['reason']],
                'similarity_scores': [r['match_score'] for r in recommendations]
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in assistant endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/experts', methods=['GET'])
def get_experts():
    """Return the parsed expert data"""
    return jsonify(experts_data)

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=8080)
