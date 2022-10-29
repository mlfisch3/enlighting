mkdir -p ~/.streamlit/

echo "\
[runner]\n\
fastReruns = false\n\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
enableXsrfProtection=false\n\
[theme]\n\
primaryColor="#dca5ce"\n\
backgroundColor="#0e1117"\n\
secondaryBackgroundColor="#31333f"\n\
textColor="#fafafa"\n\
font="sans serif"\n\
\n\
" > ~/.streamlit/config.toml