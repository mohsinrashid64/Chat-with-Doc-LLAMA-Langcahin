import streamlit as st
import requests

def main():
    st.title('Chat with Document')

    user_id = "12345"  # Replace with the actual user ID
    use_case_id = "ollama"  # Replace with the actual use case ID

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    prompt = st.chat_input("Ask Something")
    response = None
    if prompt:
        st.session_state['history'].append(("user", prompt))
        response = send_request(prompt, user_id, use_case_id)
        response = response['answer']
        st.session_state['history'].append(("ai", response))



    for speaker, text in st.session_state['history']:
        with st.chat_message(speaker, avatar=None):
            st.write(text)




def send_request(question, user_id, use_case_id):
    url = f"http://127.0.0.1:8000/chat?question={question}&user_id={user_id}&use_case_id={use_case_id}"
    response = requests.get(url)
    return response.json()["message"]

if __name__ == "__main__":
    main()
    # print('apple')