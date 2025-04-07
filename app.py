from flask import Flask, render_template, request, jsonify
from crs_calculator import CRSAdvisor
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        # Get form data
        form_data = request.form.to_dict()
        app.logger.debug(f"Raw form data: {form_data}")
        
        # Convert marital status
        form_data['status'] = 'single' if form_data.get('status') == 'single' else 'spouse'
        
        # Convert numeric fields to integers with defaults
        numeric_fields = [
            'age', 'education_level', 'canadian_education',
            'first_lang_speaking', 'first_lang_listening', 'first_lang_reading', 'first_lang_writing',
            'second_lang_speaking', 'second_lang_listening', 'second_lang_reading', 'second_lang_writing',
            'canadian_work_experience', 'foreign_work_experience',
            'spouse_education_level', 'spouse_canadian_exp',
            'spouse_lang_speaking', 'spouse_lang_listening', 'spouse_lang_reading', 'spouse_lang_writing',
            'has_job_offer'
        ]
        
        for field in numeric_fields:
            if field in form_data:
                try:
                    form_data[field] = int(form_data[field]) if form_data[field] else 0
                except ValueError:
                    form_data[field] = 0  # Default to 0 if conversion fails
            else:
                form_data[field] = 0  # Default value if field is missing
        
        # Convert boolean fields
        boolean_fields = [
            'has_sibling_in_canada', 'has_provincial_nomination',
            'clb7_and_post_secondary', 'post_secondary_and_canadian_exp',
            'foreign_and_canadian_exp', 'trade_certificate'
        ]
        
        for field in boolean_fields:
            if field in form_data:
                form_data[field] = form_data[field].lower() == 'true'
            else:
                form_data[field] = False
        
        # Handle spouse accompanying separately
        if 'spouse_accompanying' in form_data:
            form_data['spouse_accompanying'] = form_data['spouse_accompanying'].lower() == 'true'
        else:
            form_data['spouse_accompanying'] = False
        
        # Initialize advisor
        advisor = CRSAdvisor()
        
        # Calculate score
        analysis = advisor.analyze_user_inputs(form_data)
        
        # Prepare results
        results = {
            'total_score': analysis['score_breakdown']['total'],
            'breakdown': analysis['score_breakdown'],
            'suggestions': analysis['suggestions'],
            'summary': analysis['summary'],
            'form_data': form_data
        }
        
        app.logger.debug(f"Processed results: {results}")
        return render_template('results.html', results=results)
    
    except Exception as e:
        app.logger.error(f"Error processing form: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'message': 'There was an error processing your request. Please check your inputs and try again.'
        }), 400

if __name__ == '__main__':
    app.run(debug=True)