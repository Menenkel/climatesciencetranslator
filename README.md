# Climate Science Translator Assistant

A modern web application that provides expert climate science answers using ChatGPT API and connects users with specialized researchers through intelligent semantic matching.

![Climate Science Translator](https://img.shields.io/badge/Climate-Science%20Translator-blue)
![Flask](https://img.shields.io/badge/Flask-2.3.3-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5--turbo-orange)
![Python](https://img.shields.io/badge/Python-3.13+-yellow)

## 🌟 Features

### 🤖 AI-Powered Responses
- **ChatGPT Integration**: Real-time climate science answers using OpenAI's GPT-3.5-turbo
- **Context-Aware**: Considers user affiliation and thematic area for personalized responses
- **Confidence Scoring**: Each answer includes a confidence score (0-100%)

### 👥 Expert Matching System
- **Semantic Similarity**: Intelligent matching between questions and expert expertise
- **Dynamic Recommendations**: Finds relevant experts from curated database
- **Contact Information**: Direct access to expert contact details
- **Multiple Experts**: Shows up to 3 most relevant experts per question

### 🎨 Modern UI/UX
- **Dark Mode Design**: Minimalist, professional interface
- **Interactive Background**: Animated grid with hover effects
- **Smooth Transitions**: Step-by-step onboarding process
- **Responsive Design**: Works on desktop and mobile devices
- **Electric Border Effects**: Animated borders around answer boxes

### 🔄 Smart Onboarding
- **Three-Step Process**: Affiliation → Hazard → Contact Information
- **Auto-Advance**: Seamless progression through onboarding
- **Custom Input**: "Other" options with continue buttons
- **Back Navigation**: Easy correction of previous answers

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Menenkel/climatesciencetranslator.git
   cd climatesciencetranslator
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open in browser**
   ```
   http://localhost:8080
   ```

## 📁 Project Structure

```
climatesciencetranslator/
├── app.py                          # Flask backend application
├── static/
│   └── index.html                  # Frontend interface
├── fake_climate_researchers.csv    # Expert database
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (create this)
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Expert Database
The application uses `fake_climate_researchers.csv` containing:
- **Name**: Expert's full name
- **Expertise A**: Primary expertise area
- **Expertise B**: Secondary expertise area  
- **Expertise C**: Tertiary expertise area
- **Email**: Contact email address

## 🎯 Usage

### 1. Onboarding Process
1. **Select Affiliation**: Choose from predefined options or specify "Other"
2. **Choose Hazard**: Select primary hazard of interest or specify "Other"
3. **Provide Contact**: Enter your contact information

### 2. Ask Questions
- Type any climate science question
- Receive AI-generated answers with confidence scores
- Get expert recommendations with contact information

### 3. Expert Matching
The system automatically:
- Analyzes your question for key terms
- Matches against expert expertise areas
- Provides 100% scores for exact matches
- Shows multiple relevant experts when available

## 🔌 API Endpoints

### GET `/api/experts`
Returns the list of available climate science experts.

**Response:**
```json
[
  {
    "id": "e13",
    "name": "Jordan Walker",
    "affiliation": "Academia",
    "expertise_tags": "Anticipatory Action, Hydrology, Oceanography",
    "contact_email": "james88@example.com"
  }
]
```

### POST `/api/assistant`
Processes climate science questions and returns answers with expert recommendations.

**Request:**
```json
{
  "onboarding": {
    "affiliation": "academia",
    "thematic_area": "multi-hazard",
    "contact": "user@example.com"
  },
  "question": "What is anticipatory action?"
}
```

**Response:**
```json
{
  "answer": "Anticipatory action refers to...",
  "confidence": 48,
  "recommended_experts": [
    {
      "name": "Jordan Walker",
      "match_score": 100,
      "expertise_summary": "Anticipatory Action, Hydrology, Oceanography",
      "contact_info": "Contact: Jordan Walker at james88@example.com"
    }
  ],
  "follow_up": ""
}
```

## 🎨 UI Features

### Visual Elements
- **Animated Background**: Interactive grid with colorful hover effects
- **Flip Word Effect**: Dynamic headline cycling through roles (Translator, Engineer, Practitioner, etc.)
- **Electric Borders**: Animated gradient borders around answer boxes
- **Custom Scrollbars**: Styled scrollbars matching the dark theme

### Navigation
- **Top Menu**: About, Contact, FAQ links
- **Back Buttons**: Easy navigation between steps
- **Continue Buttons**: For custom "Other" selections

## 🔍 Expert Matching Algorithm

The system uses a sophisticated semantic similarity algorithm:

1. **Keyword Extraction**: Identifies climate science terms from questions
2. **Exact Matching**: 100% scores for exact word matches
3. **Partial Matching**: Handles variations and related terms
4. **Affiliation Bonus**: Small boost for matching user affiliations
5. **Dynamic Scoring**: Adjusts based on expertise relevance

### Supported Expertise Areas
- Anticipatory Action
- Climate Modeling
- Atmospheric Science
- Disaster Risk Finance
- Environmental Economics
- And 50+ more climate science areas

## 🛠️ Development

### Running in Development Mode
```bash
python app.py
```
The application runs on `http://localhost:8080` with auto-reload enabled.

### Adding New Experts
1. Edit `fake_climate_researchers.csv`
2. Add new rows with: Name, Expertise A, Expertise B, Expertise C, Email
3. Restart the application

### Customizing the Matching Algorithm
Modify the `climate_terms` list in `app.py` to add new expertise keywords.

## 🚀 Deployment

### Netlify Deployment
1. Build the frontend (if needed)
2. Connect your GitHub repository to Netlify
3. Set environment variables in Netlify dashboard
4. Deploy automatically on push

### Other Platforms
The application can be deployed on any platform supporting Python Flask applications.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for providing the ChatGPT API
- The climate science community for expertise areas
- Flask framework for the backend
- Modern web technologies for the frontend

## 📞 Support

For support, please open an issue on GitHub or contact the development team.

---

**Built with ❤️ for the climate science community**
