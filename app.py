import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from your_movie_recommender import get_movie_recommendations

load_dotenv()

def main():
    st.title("Movie Recommender App")
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'input_key' not in st.session_state:
        st.session_state['input_key'] = 0

    user_input = st.text_input("Enter your movie preferences:", key=f"user_input_{st.session_state['input_key']}")

    if st.button("Send"):
        st.session_state['history'].append(HumanMessage(content=user_input))
        
        ai_response_content = get_movie_recommendations(user_input)
        ai_response = AIMessage(content=ai_response_content)
        st.session_state['history'].append(ai_response)
        
        st.session_state['input_key'] += 1

    for index, message in enumerate(st.session_state['history']):
        if isinstance(message, HumanMessage):
            st.text_area("You:", value=message.content, height=100, disabled=True, key=f"user_{index}")
        elif isinstance(message, AIMessage):
            st.text_area("AI:", value=message.content, height=100, disabled=True, key=f"ai_{index}")

if __name__ == "__main__":
    main()