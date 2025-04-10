import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re

class CRSAdvisor:
    def __init__(self, synthetic_data_path='crs_synthetic_data.csv'):
        try:
            self.data = pd.read_csv(synthetic_data_path)
            # Ensure all expected columns are present
            required_columns = ['age', 'education_level', 'canadian_education', 
                              'first_lang_speaking', 'first_lang_listening', 
                              'first_lang_reading', 'first_lang_writing',
                              'canadian_work_experience', 'foreign_work_experience',
                              'spouse_accompanying', 'crs_score']
            for col in required_columns:
                if col not in self.data.columns:
                    raise ValueError(f"Missing required column in synthetic data: {col}")
        except FileNotFoundError:
            print(f"Warning: Could not find data file at {synthetic_data_path}")
            self.data = None
        except Exception as e:
            print(f"Warning: Error loading synthetic data: {str(e)}")
            self.data = None
        
        # Define maximum possible points for each category
        self.max_points = {
            'core': {
                'single': 500 + 150 + 136 + 80,  # Age + Education + Language + Work
                'with_spouse': 460 + 140 + 128 + 70
            },
            'age': {
                'single': 110,
                'with_spouse': 100
            },
            'education': {
                'single': 150,
                'with_spouse': 140
            },
            'language': {
                'single': 136,  # First language (34*4)
                'with_spouse': 128  # First language (32*4)
            },
            'second_language': {
                'all': 22  # Same for both categories
            },
            'canadian_work': {
                'single': 80,
                'with_spouse': 70
            },
            'foreign_work': {
                'all': 50  # From skill transferability
            },
            'spouse': {
                'education': 10,
                'language': 20,
                'work': 10,
                'total': 40
            },
            'skill_transfer': {
                'all': 100  # Combined from education, foreign work, certificates
            },
            'additional': {
                'sibling': 15,
                'french': 50,
                'canadian_edu': 30,
                'job_offer': {
                    'noc_00': 200,
                    'other': 50
                },
                'provincial_nomination': 600,
                'total': 895  # Sum of all possible additional points
            },
            'total': 1200
        }
        
        self.suggestion_templates = {
            'age': [
                ("Your age is giving you maximum points already.", 90),
                ("Consider applying soon as you'll start losing age points after 30.", 70),
                ("Your age is costing you significant points. Consider other factors to compensate.", 30)
            ],
            'education': [
                ("Your education level is already giving you maximum points.", 90),
                ("Consider upgrading your education to a higher degree for more points.", 70),
                ("Improving your education level could significantly boost your score.", 30)
            ],
            'language': [
                ("Your language scores are excellent. No improvement needed here.", 90),
                ("Improving your language scores could give you additional points.", 70),
                ("Your language proficiency is significantly reducing your score. Consider language training.", 30)
            ],
            'experience': [
                ("Your work experience is giving you maximum points.", 90),
                ("Gaining more Canadian work experience would increase your score.", 70),
                ("You have little work experience. This is hurting your score significantly.", 30)
            ],
            'canadian_connection': [
                ("You have strong Canadian connections (education/work/sibling).", 90),
                ("Consider gaining Canadian education or work experience for more points.", 70),
                ("Lack of Canadian connections is reducing your score. Explore options to gain Canadian experience.", 30)
            ],
            'spouse': [
                ("Your spouse's qualifications are contributing well to your score.", 90),
                ("Improving your spouse's language scores or education could help.", 70),
                ("Your spouse's qualifications are not contributing much. Consider ways to improve them.", 30)
            ]
        }
        
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.stop_words = set(['the', 'and', 'to', 'of', 'in', 'for', 'is', 'on', 'that', 'by', 'this', 'with', 'you', 'it', 'or', 'are', 'be', 'at', 'as', 'was', 'were', 'your'])
    
    def _select_template(self, category, percent):
        """Select the most appropriate suggestion template based on score percentage"""
        for template, threshold in self.suggestion_templates[category]:
            if percent >= threshold:
                return template
        return self.suggestion_templates[category][-1][0]  # Return the last one if none match
    
    def calculate_score_breakdown(self, user_data):
        marital_status = 'single' if user_data.get('status') == 'single' else 'with_spouse'
        
        # Calculate language scores based on CLB levels
        first_lang_score = self._calculate_language_points(
            user_data.get('first_lang_speaking', 0),
            user_data.get('first_lang_listening', 0),
            user_data.get('first_lang_reading', 0),
            user_data.get('first_lang_writing', 0),
            marital_status
        )
        
        second_lang_score = self._calculate_second_language_points(
            user_data.get('second_lang_speaking', 0),
            user_data.get('second_lang_listening', 0),
            user_data.get('second_lang_reading', 0),
            user_data.get('second_lang_writing', 0)
        )
        
        # Calculate Canadian work experience points
        canadian_work_years = user_data.get('canadian_work_experience', 0)
        canadian_work_score = self._calculate_canadian_work_points(canadian_work_years, marital_status)
        
        # Calculate foreign work experience points
        foreign_work_years = user_data.get('foreign_work_experience', 0)
        foreign_work_score = self._calculate_foreign_work_points(foreign_work_years)
        
        # Calculate age points
        age = user_data.get('age', 0)
        age_score = self._calculate_age_points(age, marital_status)
        
        # Calculate education points
        education_level = user_data.get('education_level', 1)
        education_score = self._calculate_education_points(education_level, marital_status)
        
        # Calculate Canadian education points
        canadian_education_level = user_data.get('canadian_education', 0)
        canadian_education_score = self._calculate_canadian_education_points(canadian_education_level)
        
        # Calculate spouse points if applicable
        spouse_score = 0
        if marital_status == 'with_spouse' and user_data.get('spouse_accompanying', False):
            spouse_score = self._calculate_spouse_points(
                user_data.get('spouse_education_level', 1),
                user_data.get('spouse_canadian_exp', 0),
                user_data.get('spouse_lang_speaking', 0),
                user_data.get('spouse_lang_listening', 0),
                user_data.get('spouse_lang_reading', 0),
                user_data.get('spouse_lang_writing', 0)
            )
        
        # Calculate additional points
        additional_points = self._calculate_additional_points(
            user_data.get('has_sibling_in_canada', False),
            user_data.get('has_job_offer', 0),
            user_data.get('has_provincial_nomination', False)
        )
        
        # Calculate skill transferability points
        skill_transfer_score = self._calculate_skill_transfer_points(
            user_data.get('clb7_and_post_secondary', False),
            user_data.get('post_secondary_and_canadian_exp', False),
            user_data.get('foreign_and_canadian_exp', False),
            user_data.get('trade_certificate', False)
        )
        
        breakdown = {
            'age': age_score,
            'education': education_score,
            'language': first_lang_score,
            'second_language': second_lang_score,
            'canadian_work': canadian_work_score,
            'foreign_work': foreign_work_score,
            'canadian_education': canadian_education_score,
            'spouse': {
                'education': self._calculate_spouse_education_points(user_data.get('spouse_education_level', 1)),
                'language': self._calculate_spouse_language_points(
                    user_data.get('spouse_lang_speaking', 0),
                    user_data.get('spouse_lang_listening', 0),
                    user_data.get('spouse_lang_reading', 0),
                    user_data.get('spouse_lang_writing', 0)
                ),
                'work': self._calculate_spouse_work_points(user_data.get('spouse_canadian_exp', 0))
            },
            'additional': {
                'sibling': 15 if user_data.get('has_sibling_in_canada', False) else 0,
                'french': 0,  # Not implemented in this version
                'canadian_education': canadian_education_score,
                'job_offer': self._calculate_job_offer_points(user_data.get('has_job_offer', 0)),
                'provincial_nomination': 600 if user_data.get('has_provincial_nomination', False) else 0
            },
            'skill_transfer': skill_transfer_score
        }
        
        # Calculate totals
        breakdown['core_total'] = (
            breakdown['age'] + 
            breakdown['education'] + 
            breakdown['language'] + 
            breakdown['second_language'] + 
            breakdown['canadian_work']
        )
        
        breakdown['spouse_total'] = (
            breakdown['spouse']['education'] + 
            breakdown['spouse']['language'] + 
            breakdown['spouse']['work']
        ) if marital_status == 'with_spouse' and user_data.get('spouse_accompanying', False) else 0
        
        breakdown['additional_total'] = (
            breakdown['additional']['sibling'] +
            breakdown['additional']['french'] +
            breakdown['additional']['canadian_education'] +
            breakdown['additional']['job_offer'] +
            breakdown['additional']['provincial_nomination']
        )
        
        breakdown['total'] = (
            breakdown['core_total'] + 
            breakdown['spouse_total'] + 
            breakdown['skill_transfer'] + 
            breakdown['additional_total']
        )
        
        return breakdown
    
    def _calculate_age_points(self, age, marital_status):
        """Calculate age points based on age and marital status"""
        age_points = {
            "single": [0, 99, 105] + [110]*10 + [105, 99, 94, 88, 83, 77, 72, 66, 61, 55, 50, 39, 28, 17, 6, 0],
            "with_spouse": [0, 90, 95] + [100]*10 + [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 35, 25, 15, 5, 0]
        }
        age_brackets = list(range(17, 46))
        if age < 17 or age > 45:
            return 0
        return age_points[marital_status][age_brackets.index(age)]
    
    def _calculate_education_points(self, education_level, marital_status):
        """Calculate education points based on level and marital status"""
        edu_points = {
            "single": [0, 30, 90, 98, 120, 128, 135, 150],
            "with_spouse": [0, 28, 84, 91, 112, 119, 126, 140]
        }
        return edu_points[marital_status][education_level - 1]
    
    def _calculate_canadian_education_points(self, education_level):
        """Calculate Canadian education points"""
        if education_level == 0:
            return 0
        elif education_level == 1:
            return 0
        elif education_level == 2:
            return 15
        elif education_level == 3:
            return 30
        return 0
    
    def _calculate_language_points(self, speaking, listening, reading, writing, marital_status):
        """Calculate first official language points"""
        total = 0
        scores = [speaking, listening, reading, writing]
        for clb in scores:
            if clb >= 10:
                score = 34 if marital_status == 'single' else 32
            elif clb == 9:
                score = 31 if marital_status == 'single' else 29
            elif clb == 8:
                score = 23 if marital_status == 'single' else 22
            elif clb == 7:
                score = 17 if marital_status == 'single' else 16
            elif clb == 6:
                score = 9 if marital_status == 'single' else 8
            elif clb in [4, 5]:
                score = 6
            else:
                score = 0
            total += score
        return total
    
    def _calculate_second_language_points(self, speaking, listening, reading, writing):
        """Calculate second official language points"""
        # If all scores are 0, assume no second language test was taken
        if speaking == 0 and listening == 0 and reading == 0 and writing == 0:
            return 0
            
        # Validate scores are within CLB range (1-12)
        for score in [speaking, listening, reading, writing]:
            if score < 0 or score > 12:
                return 0
                
        total = 0
        scores = [speaking, listening, reading, writing]
        for clb in scores:
            if clb <= 4:
                score = 0
            elif clb in [5, 6]:
                score = 1
            elif clb in [7, 8]:
                score = 3
            else:  # CLB 9 or more
                score = 6
            total += score
        return min(total, 22)
    
    def _calculate_canadian_work_points(self, years, marital_status):
        """Calculate Canadian work experience points"""
        can_exp_points = [0, 40, 53, 64, 72, 80] if marital_status == 'single' else [0, 35, 46, 56, 63, 70]
        return can_exp_points[min(years, 5)]
    
    def _calculate_foreign_work_points(self, years):
        """Calculate foreign work experience points"""
        if years in [1, 2]:
            return 13
        elif years >= 3:
            return 25
        return 0
    
    def _calculate_additional_points(self, has_sibling, job_offer, provincial_nomination):
        """Calculate additional points"""
        points = 0
        if has_sibling:
            points += 15
        if job_offer == 1:
            points += 200
        elif job_offer == 2:
            points += 50
        if provincial_nomination:
            points += 600
        return points
    
    def _calculate_skill_transfer_points(self, clb7_plus, edu_work, foreign_canadian, trade_cert):
        """Calculate skill transferability points"""
        points = 0
        if clb7_plus:
            points += 25
        if edu_work:
            points += 25
        if foreign_canadian:
            points += 50
        if trade_cert:
            points += 50
        return min(points, 100)
    
    def _calculate_spouse_points(self, education_level, work_years, speaking, listening, reading, writing):
        """Calculate total spouse points"""
        edu_points = [0, 2, 6, 7, 8, 9, 10, 10]
        edu_score = edu_points[education_level - 1]
        
        work_exp_score = [0, 5, 7, 8, 9, 10]
        work_score = work_exp_score[min(work_years, 5)]
        
        lang_score = 0
        for clb in [speaking, listening, reading, writing]:
            if clb >= 9:
                score = 5
            elif clb in [7, 8]:
                score = 3
            elif clb in [5, 6]:
                score = 1
            else:
                score = 0
            lang_score += score
        
        return edu_score + work_score + lang_score
    
    def _calculate_spouse_education_points(self, education_level):
        """Calculate spouse education points"""
        edu_points = [0, 2, 6, 7, 8, 9, 10, 10]
        return edu_points[education_level - 1]
    
    def _calculate_spouse_language_points(self, speaking, listening, reading, writing):
        """Calculate spouse language points"""
        total = 0
        for clb in [speaking, listening, reading, writing]:
            if clb >= 9:
                score = 5
            elif clb in [7, 8]:
                score = 3
            elif clb in [5, 6]:
                score = 1
            else:
                score = 0
            total += score
        return total
    
    def _calculate_spouse_work_points(self, work_years):
        """Calculate spouse work experience points"""
        work_exp_score = [0, 5, 7, 8, 9, 10]
        return work_exp_score[min(work_years, 5)]
    
    def _calculate_job_offer_points(self, job_offer_type):
        """Calculate job offer points"""
        if job_offer_type == 1:
            return 200
        elif job_offer_type == 2:
            return 50
        return 0
    
    def get_category_score_percentage(self, category_score, max_possible):
        return min(100, (category_score / max_possible) * 100) if max_possible > 0 else 0
    
    def calculate_impact_score(self, user_data):
        """Calculate impact scores for each category based on gap and weight"""
        marital_status = 'single' if user_data.get('status') == 'single' else 'with_spouse'
        breakdown = self.calculate_score_breakdown(user_data)
        
        # Define weights for each category (based on importance)
        weights = {
            'provincial_nomination': 10,
            'job_offer': 8,
            'canadian_work': 7,
            'language': 9,
            'education': 6,
            'age': 5,
            'second_language': 4,
            'foreign_work': 5,
            'spouse_factors': 6,
            'canadian_education': 6,
            'french': 5,
            'sibling': 3
        }
        
        impact_scores = {}
        
        # Provincial Nomination
        gap = self.max_points['additional']['provincial_nomination'] - breakdown['additional']['provincial_nomination']
        impact_scores['provincial_nomination'] = gap * weights['provincial_nomination']
        
        # Job Offer
        max_job_offer = max(self.max_points['additional']['job_offer'].values())
        gap = max_job_offer - breakdown['additional']['job_offer']
        impact_scores['job_offer'] = gap * weights['job_offer']
        
        # Canadian Work Experience
        gap = self.max_points['canadian_work'][marital_status] - breakdown['canadian_work']
        impact_scores['canadian_work'] = gap * weights['canadian_work']
        
        # Language (First Official)
        gap = self.max_points['language'][marital_status] - breakdown['language']
        impact_scores['language'] = gap * weights['language']
        
        # Education
        gap = self.max_points['education'][marital_status] - breakdown['education']
        impact_scores['education'] = gap * weights['education']
        
        # Age
        gap = self.max_points['age'][marital_status] - breakdown['age']
        impact_scores['age'] = gap * weights['age']
        
        # Second Language
        gap = self.max_points['second_language']['all'] - breakdown['second_language']
        impact_scores['second_language'] = gap * weights['second_language']
        
        # Foreign Work Experience
        gap = self.max_points['foreign_work']['all'] - breakdown['foreign_work']
        impact_scores['foreign_work'] = gap * weights['foreign_work']
        
        # Spouse Factors (if applicable)
        if marital_status == 'with_spouse' and user_data.get('spouse_accompanying', False):
            spouse_gap = (
                (self.max_points['spouse']['education'] - breakdown['spouse']['education']) +
                (self.max_points['spouse']['language'] - breakdown['spouse']['language']) +
                (self.max_points['spouse']['work'] - breakdown['spouse']['work'])
            )
            impact_scores['spouse_factors'] = spouse_gap * weights['spouse_factors']
        
        # Canadian Education
        gap = self.max_points['additional']['canadian_edu'] - breakdown['additional']['canadian_education']
        impact_scores['canadian_education'] = gap * weights['canadian_education']
        
        # French (not implemented in this version)
        gap = self.max_points['additional']['french'] - breakdown['additional']['french']
        impact_scores['french'] = gap * weights['french']
        
        # Sibling
        gap = self.max_points['additional']['sibling'] - breakdown['additional']['sibling']
        impact_scores['sibling'] = gap * weights['sibling']
        
        return impact_scores
    
    def generate_suggestions(self, user_data):
        marital_status = 'single' if user_data.get('status') == 'single' else 'with_spouse'
        breakdown = self.calculate_score_breakdown(user_data)
        impact_scores = self.calculate_impact_score(user_data)

        # Sort impact scores to get top 3 most impactful areas
        top_impact_areas = sorted(impact_scores.items(), key=lambda x: x[1], reverse=True)[:3]

        suggestions = []
        point_improvements = {}
        
        # Generate suggestions for top impact areas
        for area, impact in top_impact_areas:
            if area == 'provincial_nomination':
                suggestion = "👉 Obtaining a provincial nomination could add 600 points"
                point_improvements['provincial_nomination'] = 600
            elif area == 'job_offer':
                suggestion = "👉 Getting a NOC 00 job offer could add 200 points (or 50 points for other NOC 0/A/B jobs)"
                point_improvements['job_offer'] = 200
            elif area == 'canadian_work':
                max_points = self.max_points['canadian_work'][marital_status]
                current = breakdown['canadian_work']
                gap = max_points - current
                suggestion = f"👉 Gaining more Canadian work experience could add up to {gap} more points"
                point_improvements['canadian_work'] = gap
            elif area == 'language':
                max_points = self.max_points['language'][marital_status]
                current = breakdown['language']
                gap = max_points - current
                suggestion = f"👉 Improving your language test scores could add up to {gap} more points"
                point_improvements['language'] = gap
            elif area == 'education':
                max_points = self.max_points['education'][marital_status]
                current = breakdown['education']
                gap = max_points - current
                suggestion = f"👉 Pursuing higher education could add up to {gap} more points"
                point_improvements['education'] = gap
            elif area == 'second_language':
                max_points = self.max_points['second_language']['all']
                current = breakdown['second_language']
                gap = max_points - current
                suggestion = f"👉 Learning/improving your second official language could add up to {gap} more points"
                point_improvements['second_language'] = gap
            elif area == 'canadian_education':
                max_points = self.max_points['additional']['canadian_edu']
                current = breakdown['additional']['canadian_education']
                gap = max_points - current
                suggestion = f"👉 Completing Canadian education could add up to {gap} more points"
                point_improvements['canadian_education'] = gap
            elif area == 'french':
                max_points = self.max_points['additional']['french']
                current = breakdown['additional']['french']
                gap = max_points - current
                suggestion = f"👉 Improving French language skills could add up to {gap} more points"
                point_improvements['french'] = gap
            elif area == 'spouse_factors' and marital_status == 'with_spouse' and user_data.get('spouse_accompanying', False):
                max_points = (
                    self.max_points['spouse']['education'] +
                    self.max_points['spouse']['language'] +
                    self.max_points['spouse']['work']
                )
                current = (
                    breakdown['spouse']['education'] +
                    breakdown['spouse']['language'] +
                    breakdown['spouse']['work']
                )
                gap = max_points - current
                suggestion = f"👉 Improving your spouse's qualifications could add up to {gap} more points"
                point_improvements['spouse_factors'] = gap
            else:
                continue
            
            suggestions.append(suggestion)
        
        # Add general category suggestions
        for category in ['age', 'education', 'language', 'experience', 'canadian_connection', 'spouse']:
            if category == 'canadian_connection':
                score = (
                    breakdown['canadian_work'] + 
                    breakdown['additional']['canadian_education'] + 
                    breakdown['additional']['sibling']
                )
                max_score = (
                    self.max_points['canadian_work'][marital_status] + 
                    self.max_points['additional']['canadian_edu'] + 
                    self.max_points['additional']['sibling']
                )
            elif category == 'spouse' and (marital_status != 'with_spouse' or not user_data.get('spouse_accompanying', False)):
                continue
            else:
                score = breakdown.get(category, 0)
                max_score = self.max_points.get(category, {}).get(marital_status, 0) or self.max_points.get(category, {}).get('all', 0)
            
            percent = self.get_category_score_percentage(score, max_score)
            suggestion_text = self._select_template(category, percent)
            
            if percent < 90:  # Only suggest improvements if not near max
                potential_gain = max_score - score
                if potential_gain > 0:
                    point_improvements[category] = potential_gain
                    suggestion_text += f" (Potential gain: {potential_gain} points)"
            
            suggestions.append(suggestion_text)
        
        # Add similar profiles suggestions if synthetic data is available
        if self.data is not None:
                similar_suggestions = self._find_similar_profiles(user_data, breakdown['total'], n=3)
                suggestions.extend(similar_suggestions[:4])  # 1 header + top 3 profiles

        # Potential future score
        if point_improvements:
                potential_score = breakdown['total'] + sum(point_improvements.values())
                suggestions.append(f"By implementing these improvements, your score could potentially reach: {min(potential_score, 1200)} points (Max CRS is 1200)")

        # ✅ Add debug prints here
        print("Suggestions generated:")
        for s in suggestions:
            print(s)
        
       
        return suggestions



    def _find_similar_profiles(self, user_data, user_score, n=3):
        """Find similar profiles in synthetic data and their improvement strategies"""
        profile_text = self._create_profile_text(user_data)
        
        # Vectorize the user profile and all synthetic profiles
        all_profiles = [profile_text] + [self._create_profile_text(row) for _, row in self.data.iterrows()]
        tfidf_matrix = self.vectorizer.fit_transform(all_profiles)
        
        # Calculate cosine similarity between user and all synthetic profiles
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        # Get indices of most similar profiles
        similar_indices = similarities.argsort()[-n:][::-1]
        
        suggestions = ["Based on similar profiles, consider these strategies:"]
        
        for idx in similar_indices:
            similar_score = self.data.iloc[idx]['crs_score']
            strategy = self._generate_strategy_from_profile(self.data.iloc[idx], user_score)
            suggestions.append(f"- Profile with score {similar_score}: {strategy}")
        
        return suggestions
    
    def _create_profile_text(self, profile):
        """Create a text representation of a profile for similarity comparison"""
        if isinstance(profile, dict):
            text_parts = [
                f"Age {profile.get('age', 0)}",
                f"Education level {profile.get('education_level', 0)}",
                f"Canadian education {profile.get('canadian_education', 0)}",
                f"Language scores {profile.get('lang_score', 0)}",
                f"Canadian experience {profile.get('canadian_work_experience', 0)} years",
                f"Foreign experience {profile.get('foreign_work_experience', 0)} years",
                f"Spouse {'present' if profile.get('spouse_accompanying', False) else 'absent'}"
            ]
        else:
            text_parts = [
                f"Age {profile['age']}",
                f"Education level {profile['education_level']}",
                f"Canadian education {profile['canadian_education']}",
                f"Language scores {sum([profile[f'first_lang_{a}'] for a in ['speaking', 'listening', 'reading', 'writing']])}",
                f"Canadian experience {profile['canadian_work_experience']} years",
                f"Foreign experience {profile['foreign_work_experience']} years",
                f"Spouse {'present' if profile['spouse_accompanying'] else 'absent'}"
            ]
        return " ".join(text_parts)
    
    def _generate_strategy_from_profile(self, profile, user_score):
        """Generate strategy text from a similar profile"""
        strategy = []
        if profile['crs_score'] > user_score:
            if profile['canadian_education'] > 0:
                strategy.append("completed Canadian education")
            if profile['canadian_work_experience'] > 2:
                strategy.append("gained Canadian work experience")
            if any(profile[f'first_lang_{a}'] >= 9 for a in ['speaking', 'listening', 'reading', 'writing']):
                strategy.append("improved language scores")
            if profile['has_provincial_nomination']:
                strategy.append("obtained provincial nomination")
        return "Successfully " + ", ".join(strategy) if strategy else "Focused on multiple factors"

    def analyze_user_inputs(self, user_inputs):
        breakdown = self.calculate_score_breakdown(user_inputs)
        suggestions = self.generate_suggestions(user_inputs)
        summary = self._create_summary(user_inputs, suggestions)
        
        return {
            'score_breakdown': breakdown,
            'suggestions': suggestions,
            'summary': summary,
            'impact_scores': self.calculate_impact_score(user_inputs)
        }
    
    def _create_summary(self, user_inputs, suggestions):
        """Create a summary of key improvement areas"""
        words = re.findall(r'\b[a-z]{3,}\b', " ".join(suggestions).lower())
        word_freq = defaultdict(int)
        for word in words:
            if word not in self.stop_words:
                word_freq[word] += 1
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        return f"Key areas to focus on: {', '.join([w[0] for w in top_words])}"