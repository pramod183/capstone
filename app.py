import streamlit as st
from TweetSearch import TweetSearch  # Assuming TweetSearch is defined in a separate file

def main():
    st.title('Sentiment Analysis')
    st.write('Please enter keywords for tweet search:')

    keys = st.text_input('Enter Keywords for tweets search:')
    
    if st.button('Search Tweets'):
        # Call the search method from TweetSearch class
        tweetset = TweetSearch.search(keys)
        # Display the search results
        if tweetset:
            st.write('Search Results:')
            for tweet in tweetset:
                st.write('- ' + tweet)
        else:
            st.write('No tweets found for the given keywords.')

if __name__ == '__main__':
    main()
