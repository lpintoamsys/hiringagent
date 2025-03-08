# main.py
import streamlit as st
import pandas as pd
from utils.utils import (
    ingest_inputs,
    parse_job_description,
    parse_resumes,
    score_candidates,
    rank_candidates,
    generate_email_templates,
)
import asyncio
import plotly.graph_objects as go



# Main App Title and Description
st.title("Resume Screening Agent")

# Input section for job description
st.header("Job Description Input")
job_description = st.text_area("Paste the job description or URL", height=150)

# Input section for candidate resumes
st.header("Candidate Resumes")
resume_files = st.file_uploader(
    "Upload resume files (PDF/Word)",
    type=["pdf", "docx", "doc"],
    accept_multiple_files=True,
)

st.header("Candidates to Select")
num_candidates = st.slider(
    "Select the number of candidates to invite for an interview", 1, 4, 2
)


# Button to trigger the agent
if st.button("Run Agent"):
    if not job_description:
        st.error("Please provide a job description or URL.")
    elif not resume_files:
        st.error("Please upload at least one resume file.")
    else:
        st.markdown("### Your AI Agent is now processing your inputs...")
        status_text = st.empty()  # placeholder for status updates

        # Step 1: processing resumes
        with st.spinner("Step 1: Processing Inputs..."):
            # raw_data = ingest_inputs(job_description, resume_files)
            raw_data = asyncio.run(ingest_inputs(job_description, resume_files))
            status_text.text("Step 1 complete: Inputs processed.")
            with st.expander("View Processed Inputs", expanded=False):
                st.json(raw_data)

        # Step 2: processing Job description
        with st.spinner("Step 2: Processing Job Description & Resume..."):
            parsed_requirements = asyncio.run(parse_job_description(raw_data))
            parsed_resumes = asyncio.run(parse_resumes(resume_files))
            status_text.text("Step 2 complete: Job description & Resume processed.")
            with st.expander("View Parsed Job Description", expanded=False):
                st.json(parsed_requirements)
            with st.expander("View processed Resume", expanded=False):
                st.json(parsed_resumes)

        # Step 3: Score candidates based on the parsed data
        with st.spinner("Step 3: Scoring candidates..."):
            status_text.text("Step 3: Scoring candidates...")
            candidate_scores = asyncio.run(
                score_candidates(parsed_requirements, parsed_resumes)
            )
            status_text.text("Step 3 complete: Candidates scored.")
            
            # Display detailed skills analysis for each candidate
            st.markdown("### Skills Gap Analysis")
            for candidate in candidate_scores:
                with st.expander(f"Skills Analysis for {candidate['name']}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Overall Scores")
                        st.write(f"Relevance: {candidate['relevance']}%")
                        st.write(f"Experience: {candidate['experience']}%")
                        st.write(f"Skills: {candidate['skills']}%")
                        st.write(f"Overall: {candidate['overall']}%")
                    
                    if candidate.get('skill_gap_analysis'):
                        gap_analysis = candidate['skill_gap_analysis']
                        with col2:
                            st.markdown("#### Skills Match")
                            st.write(f"Overall Skills Match: {gap_analysis['overall_match_percentage']:.1f}%")
                            
                            # Create skills visualization
                            matching = len(gap_analysis['matching_skills'])
                            partial = len(gap_analysis['partial_matches'])
                            missing = len(gap_analysis['missing_skills'])
                            
                            fig = go.Figure(data=[
                                go.Bar(name='Matching', x=['Skills'], y=[matching], marker_color='green'),
                                go.Bar(name='Partial Match', x=['Skills'], y=[partial], marker_color='yellow'),
                                go.Bar(name='Missing', x=['Skills'], y=[missing], marker_color='red')
                            ])
                            fig.update_layout(barmode='stack', height=200, margin=dict(t=0, b=0))
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Display detailed skills breakdown
                        st.markdown("#### Detailed Skills Breakdown")
                        col3, col4, col5 = st.columns(3)
                        
                        with col3:
                            st.markdown("**✅ Matching Skills**")
                            for skill in gap_analysis['matching_skills']:
                                st.write(f"- {skill}")
                                
                        with col4:
                            st.markdown("**⚠️ Partial Matches**")
                            for match in gap_analysis['partial_matches']:
                                for skill, score in match.items():
                                    st.write(f"- {skill} ({score*100:.0f}%)")
                                
                        with col5:
                            st.markdown("**❌ Missing Skills**")
                            for skill in gap_analysis['missing_skills']:
                                st.write(f"- {skill}")
                        
                        # Display recommendations
                        st.markdown("#### Skill Development Recommendations")
                        for rec in gap_analysis['skill_recommendations']:
                            for skill, recommendation in rec.items():
                                st.write(f"**{skill}:** {recommendation}")
            
            with st.expander("View Resume Summaries", expanded=False):
                st.json(candidate_scores)

        # Step 4: Rank the candidates
        with st.spinner("Step 4: Ranking candidates..."):
            status_text.text("Step 4: Ranking candidates...")
            ranked_candidates = rank_candidates(candidate_scores)
            status_text.text("Step 4 complete: Candidates ranked.")
            
            # Display ranking with skills gap summary
            st.markdown("### Candidate Rankings")
            for rank, candidate in enumerate(ranked_candidates, 1):
                st.markdown(f"**{rank}. {candidate['name']} (Overall: {candidate['overall']}%)**")
                if candidate.get('skill_gap_analysis'):
                    gap = candidate['skill_gap_analysis']
                    st.write(f"Skills Match: {gap['overall_match_percentage']:.1f}% | "
                            f"Matching Skills: {len(gap['matching_skills'])} | "
                            f"Partial Matches: {len(gap['partial_matches'])} | "
                            f"Missing Skills: {len(gap['missing_skills'])}")
            
            with st.expander("View Ranked Candidates", expanded=False):
                st.json(ranked_candidates)

        # Step 5: Generate email templates for top candidates and others
        with st.spinner("Step 5: Generating email templates..."):
            status_text.text("Step 5: Generating email templates...")
            # 'num_candidates' is assumed to come from the frontend (e.g., top X candidates)
            email_templates = asyncio.run(
                generate_email_templates(
                    ranked_candidates, parsed_requirements, num_candidates
                )
            )
            status_text.text("Step 5 complete: Email templates generated.")
            with st.expander("View Email Templates", expanded=False):
                st.json(email_templates)

        # Final update
        status_text.text("Agent processing complete! Your results are ready.")