import streamlit as st
from rag import retrieve, generate_answer, LLMError




print("script ran")

st.title("EU Regulatory AI Assistant")

question = st.text_input("Ask your question:")

if st.button("Ask"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        st.write("You asked:", question)

        try:
            with st.spinner("Generating answer..."):
                answer, context = generate_answer(question)
        except LLMError:
            st.error("The language model is currently unavailable. Please try again.")
            answer = None
        if answer:
            st.subheader("Answer")
            st.write(answer)
        results = retrieve(question)


        for chunk_id, document, distance in results:
            st.subheader(f"{chunk_id}, distance {distance:.3f}")
            with st.expander(f'{chunk_id} details : '):
                st.write(document)