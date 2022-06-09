import whois

# https://github.com/richardpenman/whois
def is_domain_available(domain:str,extensions):
    query = [whois.whois(f'{domain}.{ext}') for ext in extensions]
    
    
if __name__ == "__main__":
    x = is_domain_available('focal.ai')
    print(x)