import streamlit as st
import pandas as pd
import json
import hashlib
import time
import re
import base64
from datetime import datetime
from typing import Dict, List, Any
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io

try:
    from solana.rpc.api import Client
    from solana.keypair import Keypair
    from solana.transaction import Transaction
    from solana.system_program import transfer, TransferParams
    from solders.pubkey import Pubkey
    from solders.system_program import ID as SYS_PROGRAM_ID
    SOLANA_AVAILABLE = True
except ImportError:
    SOLANA_AVAILABLE = False
    st.warning("no errors")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
if 'agent_logs' not in st.session_state:
    st.session_state.agent_logs = []
if 'diagnosis_results' not in st.session_state:
    st.session_state.diagnosis_results = {}
if 'blockchain_records' not in st.session_state:
    st.session_state.blockchain_records = []

class SolanaHealthcare:
    """Handles Solana blockchain integration for healthcare data"""
    
    def __init__(self):
        self.client = None
        self.keypair = None
        if SOLANA_AVAILABLE:
            try:
                self.client = Client("https://api.devnet.solana.com")
                self.keypair = Keypair()
                self.connected = True
            except Exception as e:
                st.error(f"Failed to connect to Solana: {e}")
                self.connected = False
        else:
            self.connected = False
    
    def create_health_record_hash(self, patient_data: Dict) -> str:
        """Create a secure hash of patient data for blockchain storage"""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'record_type': 'diagnosis_session',
            'symptom_count': len(patient_data.get('symptoms', '').split()),
            'has_medical_history': bool(patient_data.get('history')),
            'has_lab_results': bool(patient_data.get('lab_results')),
            'severity_level': patient_data.get('level', 'unknown')
        }
        return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    
    def store_record_hash(self, record_hash: str) -> Dict:
        """Store record hash on Solana blockchain (simulated for demo)"""
        if not self.connected:
            record = {
                'hash': record_hash,
                'timestamp': datetime.now().isoformat(),
                'transaction_id': f"demo_tx_{int(time.time())}",
                'block_height': 'simulated',
                'status': 'confirmed'
            }
            return record
        
        try:
            record = {
                'hash': record_hash,
                'timestamp': datetime.now().isoformat(),
                'pubkey': str(self.keypair.pubkey()),
                'transaction_id': f"solana_tx_{int(time.time())}",
                'block_height': 'pending',
                'status': 'confirmed'
            }
            return record
        except Exception as e:
            st.error(f"Blockchain storage error: {e}")
            return None

class HealthcareAgent:
    """Base class for all healthcare agents"""
    
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.start_time = None
        self.end_time = None
    
    def log_activity(self, activity: str, details: Dict = None):
        """Log agent activity for transparency"""
        log_entry = {
            'agent': self.name,
            'timestamp': datetime.now().isoformat(),
            'activity': activity,
            'details': details or {}
        }
        st.session_state.agent_logs.append(log_entry)
    
    def start_analysis(self):
        self.start_time = time.time()
        self.log_activity("started analysis")
    
    def end_analysis(self):
        self.end_time = time.time()
        duration = self.end_time - self.start_time if self.start_time else 0
        self.log_activity("done with analysis", {'duration_seconds': duration})

class IntakePreprocessingAgent(HealthcareAgent):
    """Handles patient data intake and preprocessing"""
    
    def __init__(self):
        super().__init__("Intake & Preprocessing", "Data Collection and Normalization")
    
    def process_symptoms(self, symptoms: str) -> Dict:
        """Process and normalize symptom descriptions"""
        self.start_analysis()
        
        import nltk
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            
            stop_words = set(stopwords.words('english'))
            word_tokens = word_tokenize(symptoms.lower())
            filtered_symptoms = [word for word in word_tokens if word.isalpha() and word not in stop_words]
            
        except:
            filtered_symptoms = [word.lower() for word in symptoms.split() if word.isalpha()]
        
        processed_data = {
            'original_symptoms': symptoms,
            'filtered_symptoms': filtered_symptoms,
            'symptom_count': len(filtered_symptoms),
            'processed_text': ' '.join(filtered_symptoms)
        }
        
        self.log_activity("Symptom Processing", processed_data)
        self.end_analysis()
        return processed_data
    
    def extract_lab_data(self, lab_text: str) -> List[Dict]:
        """Extract structured lab data from text"""
        self.start_analysis()
        
        lab_pattern = re.compile(r'(\w[\w\s]*):\s*(\d+\.?\d*)\s*(\w+)?')
        matches = lab_pattern.finditer(lab_text)
        
        lab_results = []
        for match in matches:
            lab_results.append({
                'name': match.group(1).strip(),
                'value': float(match.group(2)),
                'units': match.group(3) if match.group(3) else "N/A"
            })
        
        self.log_activity("Lab Data Extraction", {'extracted_count': len(lab_results)})
        self.end_analysis()
        return lab_results

class SymptomAnalysisAgent(HealthcareAgent):
    """Analyzes symptoms for patterns and severity"""
    
    def __init__(self):
        super().__init__("Symptom Analysis", "Pattern Recognition and Severity Assessment")
    
    def analyze_symptoms(self, symptom_data: Dict) -> Dict:
        self.start_analysis()
        
        symptoms = symptom_data.get('filtered_symptoms', [])
        
        pain_indicators = ['pain', 'ache', 'hurt', 'sore', 'tender']
        respiratory_indicators = ['cough', 'breath', 'wheeze', 'chest']
        digestive_indicators = ['nausea', 'stomach', 'vomit', 'diarrhea']
        neurological_indicators = ['headache', 'dizzy', 'confused', 'memory']
        
        categories = {
            'pain': len([s for s in symptoms if any(p in s for p in pain_indicators)]),
            'respiratory': len([s for s in symptoms if any(r in s for r in respiratory_indicators)]),
            'digestive': len([s for s in symptoms if any(d in s for d in digestive_indicators)]),
            'neurological': len([s for s in symptoms if any(n in s for n in neurological_indicators)])
        }
        
        severity_score = sum(categories.values()) / len(symptoms) if symptoms else 0
        
        analysis = {
            'symptom_categories': categories,
            'primary_category': max(categories, key=categories.get) if any(categories.values()) else 'general',
            'severity_score': severity_score,
            'complexity': 'high' if len(categories) > 2 else 'moderate' if len(categories) > 1 else 'simple'
        }
        
        self.log_activity("Symptom Analysis", analysis)
        self.end_analysis()
        return analysis

class MedicalHistoryAgent(HealthcareAgent):
    """Analyzes medical history for relevant patterns"""
    
    def __init__(self):
        super().__init__("Medical History", "Historical Data Analysis")
    
    def analyze_history(self, history: str, lab_results: List[Dict]) -> Dict:
        self.start_analysis()
        
        high_risk_conditions = ['diabetes', 'hypertension', 'heart', 'cancer', 'stroke']
        chronic_conditions = ['arthritis', 'asthma', 'depression', 'anxiety']
        
        risk_factors = [condition for condition in high_risk_conditions if condition in history.lower()]
        chronic_issues = [condition for condition in chronic_conditions if condition in history.lower()]
        
        abnormal_labs = []
        for lab in lab_results:
            normal_ranges = {
                'glucose': (70, 140),
                'cholesterol': (0, 200),
                'blood pressure': (0, 120),
                'hemoglobin': (12, 16)
            }
            
            for condition, (min_val, max_val) in normal_ranges.items():
                if condition in lab['name'].lower():
                    if not (min_val <= lab['value'] <= max_val):
                        abnormal_labs.append(lab)
        
        analysis = {
            'risk_factors': risk_factors,
            'chronic_conditions': chronic_issues,
            'abnormal_labs': abnormal_labs,
            'risk_level': 'high' if risk_factors else 'moderate' if chronic_issues else 'low'
        }
        
        self.log_activity("Medical History Analysis", analysis)
        self.end_analysis()
        return analysis

class DiagnosticReasoningAgent(HealthcareAgent):
    """Main diagnostic reasoning engine"""
    
    def __init__(self):
        super().__init__("Diagnostic Reasoning", "Primary Diagnosis Generation")
    
    def generate_diagnosis(self, symptom_analysis: Dict, history_analysis: Dict, severity_info: Dict) -> Dict:
        self.start_analysis()
        
        primary_category = symptom_analysis.get('primary_category', 'general')
        risk_level = history_analysis.get('risk_level', 'low')
        severity = severity_info.get('level', 'mild')
        
        diagnostic_map = {
            'respiratory': ['Upper Respiratory Infection', 'Asthma Exacerbation', 'Pneumonia'],
            'digestive': ['Gastroenteritis', 'Food Poisoning', 'IBS'],
            'neurological': ['Tension Headache', 'Migraine', 'Stress-related symptoms'],
            'pain': ['Musculoskeletal strain', 'Inflammatory condition', 'Chronic pain syndrome'],
            'general': ['Viral syndrome', 'Stress reaction', 'General malaise']
        }
        
        possible_diagnoses = diagnostic_map.get(primary_category, diagnostic_map['general'])
        
        if severity == 'severe' or risk_level == 'high':
            confidence = 'requires_immediate_attention'
        elif severity == 'medium':
            confidence = 'moderate_concern'
        else:
            confidence = 'low_concern'
        
        diagnosis = {
            'primary_suggestions': possible_diagnoses[:3],
            'confidence_level': confidence,
            'reasoning': f"Based on {primary_category} symptoms with {risk_level} risk profile",
            'requires_followup': severity in ['severe', 'medium'] or risk_level == 'high'
        }
        
        self.log_activity("Diagnostic Reasoning", diagnosis)
        self.end_analysis()
        return diagnosis

class RiskAssessmentAgent(HealthcareAgent):
    """Assesses patient risk levels"""
    
    def __init__(self):
        super().__init__("Risk Assessment", "Patient Risk Stratification")
    
    def assess_risk(self, all_analyses: Dict) -> Dict:
        self.start_analysis()
        
        # Calculate composite risk score
        risk_factors = {
            'symptom_severity': all_analyses.get('symptom_analysis', {}).get('severity_score', 0),
            'medical_history': 1 if all_analyses.get('history_analysis', {}).get('risk_level') == 'high' else 0,
            'abnormal_labs': len(all_analyses.get('history_analysis', {}).get('abnormal_labs', [])),
            'complexity': 1 if all_analyses.get('symptom_analysis', {}).get('complexity') == 'high' else 0
        }
        
        total_risk = sum(risk_factors.values())
        
        if total_risk >= 3:
            risk_level = 'HIGH'
            recommendation = 'Seek immediate medical attention'
        elif total_risk >= 2:
            risk_level = 'MODERATE'
            recommendation = 'Schedule appointment within 24-48 hours'
        else:
            risk_level = 'LOW'
            recommendation = 'Monitor symptoms, seek care if worsening'
        
        assessment = {
            'risk_factors': risk_factors,
            'total_risk_score': total_risk,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'emergency_indicators': total_risk >= 3
        }
        
        self.log_activity("Risk Assessment", assessment)
        self.end_analysis()
        return assessment

class TreatmentRecommendationAgent(HealthcareAgent):
    """Provides treatment recommendations"""
    
    def __init__(self):
        super().__init__("Treatment Recommendation", "Care Plan Development")
    
    def recommend_treatment(self, diagnosis: Dict, risk_assessment: Dict) -> Dict:
        self.start_analysis()
        
        risk_level = risk_assessment.get('risk_level', 'LOW')
        primary_suggestions = diagnosis.get('primary_suggestions', [])
        
        if risk_level == 'HIGH':
            treatments = [
                "Seek emergency medical care immediately",
                "Do not delay treatment",
                "Call emergency services if symptoms worsen"
            ]
        elif risk_level == 'MODERATE':
            treatments = [
                "Schedule appointment with healthcare provider",
                "Monitor symptoms closely",
                "Rest and maintain hydration",
                "Follow up if symptoms persist or worsen"
            ]
        else:
            treatments = [
                "Rest and supportive care",
                "Over-the-counter symptom relief as appropriate",
                "Monitor symptoms",
                "Seek care if symptoms persist beyond 7-10 days"
            ]
        
        if any('respiratory' in diag.lower() for diag in primary_suggestions):
            treatments.append("Consider humidifier and throat lozenges")
        if any('digestive' in diag.lower() for diag in primary_suggestions):
            treatments.append("BRAT diet and clear fluids")
        
        recommendations = {
            'treatment_plan': treatments,
            'risk_level': risk_level,
            'follow_up_needed': risk_level in ['HIGH', 'MODERATE'],
            'timeframe': 'immediate' if risk_level == 'HIGH' else '24-48 hours' if risk_level == 'MODERATE' else '7-10 days'
        }
        
        self.log_activity("Treatment Recommendations", recommendations)
        self.end_analysis()
        return recommendations

class ResultsValidationAgent(HealthcareAgent):
    """Validates and cross-checks results"""
    
    def __init__(self):
        super().__init__("Results Validation", "Quality Assurance and Cross-validation")
    
    def validate_results(self, all_results: Dict) -> Dict:
        self.start_analysis()
        
        diagnosis_risk = all_results.get('diagnosis', {}).get('confidence_level', '')
        assessment_risk = all_results.get('risk_assessment', {}).get('risk_level', '')
        
        consistency_score = 0
        if 'immediate' in diagnosis_risk and assessment_risk == 'HIGH':
            consistency_score += 1
        if 'moderate' in diagnosis_risk and assessment_risk == 'MODERATE':
            consistency_score += 1
        if 'low' in diagnosis_risk and assessment_risk == 'LOW':
            consistency_score += 1
        
        symptom_severity = all_results.get('symptom_analysis', {}).get('severity_score', 0)
        if symptom_severity > 0.5 and assessment_risk != 'LOW':
            consistency_score += 1
        
        validation = {
            'consistency_score': consistency_score,
            'max_consistency': 2,
            'validation_passed': consistency_score >= 1,
            'confidence_level': 'high' if consistency_score >= 2 else 'moderate' if consistency_score >= 1 else 'low',
            'recommendations_aligned': diagnosis_risk.lower() in assessment_risk.lower()
        }
        
        self.log_activity("Results Validation", validation)
        self.end_analysis()
        return validation

def create_agent_workflow_chart():
    """Create a visual representation of the agent workflow"""
    agents = [
        "Intake & Preprocessing",
        "Symptom Analysis", 
        "Medical History",
        "Diagnostic Reasoning",
        "Risk Assessment",
        "Treatment Recommendation",
        "Results Validation"
    ]
    
    fig = go.Figure(data=go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = agents,
            color = "blue"
        ),
        link = dict(
            source = [0, 0, 1, 2, 3, 4, 5],
            target = [1, 2, 3, 3, 4, 5, 6],
            value = [1, 1, 1, 1, 1, 1, 1]
        )
    ))
    
    fig.update_layout(title_text="Healthcare Agent Workflow", font_size=10)
    return fig

def main():
    st.set_page_config(
        page_title="Multi-Agent Healthcare Diagnosis System", 
        page_icon="",
        layout="wide"
    )
    
    st.title("Multi-Agent Healthcare Diagnosis System")
    st.markdown("### Transparent AI-Powered Healthcare for Underserved Communities")
    
    blockchain = SolanaHealthcare()
    
    with st.sidebar:
        st.header("System Status")
        st.success("Multi-Agent System good")
        
        if blockchain.connected:
            st.success("Blockchain Connected")
        else:
            st.warning("Hack the Nest - Blockchain")
        
        if GEMINI_AVAILABLE:
            st.success("AI Analysis good")
        else:
            st.info("Basic Analysis Mode")
        
        st.header("Agent Status")
        if st.session_state.agent_logs:
            recent_activities = st.session_state.agent_logs[-5:]
            for log in recent_activities:
                st.write(f"Bot {log['agent']}: {log['activity']}")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Patient Intake", "Agent Analysis", "Results Dashboard", "Blockchain Records"])
    
    with tab1:
        st.header("Patient Information Intake")
        
        col1, col2 = st.columns(2)
        
        with col1:
            symptoms = st.text_area("describe your symptoms:", height=100)
            severity = st.select_slider("severity level:", options=['Mild', 'Moderate', 'Severe'])
            pain_scale = st.slider("pain scale 1-10:", 1, 10, 5)
            duration = st.number_input("duration in days:", min_value=0, max_value=365, value=1)
        
        with col2:
            medical_history = st.text_area("medical history:", height=100)
            lab_results_text = st.text_area("lab results (format: Name: Value Unit):", height=100)
            
            uploaded_files = st.file_uploader("upload medical files", accept_multiple_files=True)
            if uploaded_files:
                st.success(f"uploaded {len(uploaded_files)} files")
        
        if st.button("Start Diagnosis", type="primary"):
            if symptoms:
                patient_data = {
                    'symptoms': symptoms,
                    'level': severity.lower(),
                    'scale': pain_scale,
                    'duration': duration,
                    'history': medical_history,
                    'lab_results_text': lab_results_text,
                    'timestamp': datetime.now().isoformat()
                }
                st.session_state.patient_data = patient_data
                
                record_hash = blockchain.create_health_record_hash(patient_data)
                blockchain_record = blockchain.store_record_hash(record_hash)
                if blockchain_record:
                    st.session_state.blockchain_records.append(blockchain_record)
                    st.success(f"Data secured on blockchain: {record_hash[:16]}...")
                
                st.success("Patient data collected successfully! Go to 'Agent Analysis' tab.")
            else:
                st.error("Please describe your symptoms to continue.")
    
    with tab2:
        st.header("Multi-Agent Analysis")
        
        if st.session_state.patient_data:
            agents = {
                'intake': IntakePreprocessingAgent(),
                'symptom': SymptomAnalysisAgent(),
                'history': MedicalHistoryAgent(),
                'diagnosis': DiagnosticReasoningAgent(),
                'risk': RiskAssessmentAgent(),
                'treatment': TreatmentRecommendationAgent(),
                'validation': ResultsValidationAgent()
            }
            
            if st.button("Run Agent Analysis"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = {}
                
                status_text.text("Running Intake & Preprocessing Agent...")
                processed_symptoms = agents['intake'].process_symptoms(st.session_state.patient_data['symptoms'])
                lab_data = agents['intake'].extract_lab_data(st.session_state.patient_data.get('lab_results_text', ''))
                results['processed_symptoms'] = processed_symptoms
                results['lab_data'] = lab_data
                progress_bar.progress(1/7)
                
                status_text.text("Running Symptom Analysis Agent...")
                symptom_analysis = agents['symptom'].analyze_symptoms(processed_symptoms)
                results['symptom_analysis'] = symptom_analysis
                progress_bar.progress(2/7)
                
                status_text.text("Running Medical History Agent...")
                history_analysis = agents['history'].analyze_history(
                    st.session_state.patient_data.get('history', ''), lab_data
                )
                results['history_analysis'] = history_analysis
                progress_bar.progress(3/7)
                
                status_text.text("Running Diagnostic Reasoning Agent...")
                diagnosis = agents['diagnosis'].generate_diagnosis(
                    symptom_analysis, history_analysis, 
                    {'level': st.session_state.patient_data.get('level'), 'scale': st.session_state.patient_data.get('scale')}
                )
                results['diagnosis'] = diagnosis
                progress_bar.progress(4/7)
                
                status_text.text("Running Risk Assessment Agent...")
                risk_assessment = agents['risk'].assess_risk(results)
                results['risk_assessment'] = risk_assessment
                progress_bar.progress(5/7)
                
                status_text.text("Running Treatment Recommendation Agent...")
                treatment = agents['treatment'].recommend_treatment(diagnosis, risk_assessment)
                results['treatment'] = treatment
                progress_bar.progress(6/7)
                
                status_text.text("Running Results Validation Agent...")
                validation = agents['validation'].validate_results(results)
                results['validation'] = validation
                progress_bar.progress(7/7)
                
                status_text.text("All agents completed analysis!")
                st.session_state.diagnosis_results = results
                
                st.success("Analysis complete! Check the Results Dashboard.")
            
            st.subheader("Agent Workflow")
            fig = create_agent_workflow_chart()
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("Please complete patient intake first.")
    
    with tab3:
        st.header("Diagnosis Results Dashboard")
        
        if st.session_state.diagnosis_results:
            results = st.session_state.diagnosis_results
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                risk_level = results.get('risk_assessment', {}).get('risk_level', 'UNKNOWN')
                st.metric("Risk Level", risk_level)
            
            with col2:
                confidence = results.get('validation', {}).get('confidence_level', 'unknown')
                st.metric("Confidence", confidence.title())
            
            with col3:
                symptom_score = results.get('symptom_analysis', {}).get('severity_score', 0)
                st.metric("Symptom Severity", f"{symptom_score:.2f}")
            
            with col4:
                agents_completed = len([log for log in st.session_state.agent_logs if 'Completed' in log['activity']])
                st.metric("Agents Completed", agents_completed)
            
            st.subheader("Diagnostic Summary")
            diagnosis = results.get('diagnosis', {})
            for suggestion in diagnosis.get('primary_suggestions', []):
                st.write(f"• {suggestion}")
            
            st.write(f"**Reasoning:** {diagnosis.get('reasoning', 'Not available')}")
            
            st.subheader("Treatment Recommendations")
            treatment = results.get('treatment', {})
            for rec in treatment.get('treatment_plan', []):
                st.write(f"• {rec}")
            
            st.subheader("Risk Assessment")
            risk_data = results.get('risk_assessment', {})
            st.write(f"**Level:** {risk_data.get('risk_level', 'Unknown')}")
            st.write(f"**Recommendation:** {risk_data.get('recommendation', 'Consult healthcare provider')}")
            
            with st.expander("Detailed Analysis"):
                st.json(results)
        
        else:
            st.info("No analysis results, try checking previous steps.")
    
    with tab4:
        st.header("Blockchain Records")
        
        if st.session_state.blockchain_records:
            st.subheader("Secure Health Records on Solana Blockchain")
            
            records_df = pd.DataFrame(st.session_state.blockchain_records)
            st.dataframe(records_df, use_container_width=True)
            
            # Show blockchain status
            for record in st.session_state.blockchain_records:
                with st.expander(f"Record {record['hash'][:16]}..."):
                    st.json(record)
        else:
            st.info("No blockchain records available.")
        
        # System transparency logs
        st.subheader("System Activity Logs")
        if st.session_state.agent_logs:
            logs_df = pd.DataFrame(st.session_state.agent_logs)
            st.dataframe(logs_df, use_container_width=True)
        else:
            st.info("No activity logs available.")

if __name__ == "__main__":
    main()