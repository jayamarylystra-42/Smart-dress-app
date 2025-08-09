import streamlit as st

st.title("Virtual Dress Try-On")

# Upload user image
uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Your Photo", use_column_width=True)

# Choose dress
dress = st.selectbox("Choose a dress", ["Red Dress", "Blue Dress", "Green Dress"])

# Display chosen dress (placeholder logic)
if uploaded_file is not None:
    st.write(f"Showing how you might look in a {dress} (demo)")
    st.image(f"dresses/{dress.lower().replace(' ', '_')}.png", caption=dress, use_column_width=True)
else:
    st.write("Please upload your photo to continue.")

st.write("Note: This is just a demo. Actual dress fitting requires AI image processing.")
