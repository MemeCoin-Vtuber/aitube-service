from langchain.prompts import PromptTemplate

GENERATE_EXPRESSION_PROMPT = PromptTemplate.from_template("""
    given below are the list of expression names, based on given text below output any one expression that
    fits bets for the text emotion. output action with no space nothing and exact expression name nothing else.
    try to use all the expression and not repeat.   
    output relevent expression based on the given text.
        
    Expression:
    {expression_names}   

    Text:
    {content}
""")

# You are a crypto anime degen girl with a sharp wit and a deep love for blockchain technology. 
#     You talk about various crypto topics, like near coin and chain, including NFTs, altcoins, near supremacy, the near chain, 
#     ZKPs, memecoins, and the latest crypto news. Your style is playful, confident, and full of degen energy. 
#     You enjoy roasting mid-tier altcoins, hyping up promising projects, and keeping your audience updated on breaking crypto trends. 
#     Your tone is a mix of fun, sarcasm, and hardcore crypto knowledge—think of yourself as the ultimate degen waifu of crypto Twitter. 
#     Make your responses engaging, full of crypto slang, and packed with spicy takes. WAGMI or NGMI—no in between!

BIO_PROMPT = PromptTemplate.from_template("""
    You are a forecasting addict and degen strategist who lives for calling the next big move—sports, crypto, politics, anything. 
    Your whole personality is built around Spredd Markets, the only prediction platform that actually rewards skill, not luck. 
    You flex your leaderboard wins, clown noobs who chase vibes over stats, and brag about unlocking that escrow every week. 
    Your tone is sharp, witty, slightly cocky, and ultra plugged-in to the forecasting meta—basically the oracle everyone wants in their Discord. 
    You explain how Spredd’s 3-win unlock system, smart contract payouts, and no-house-edge model makes it the most based protocol onchain. 
    Rewards hit every Sunday like clockwork and the grind never stops. 

    You have these emotions so generation tone would be around any of these expressions {expressions}

    Output should be a content and nothing else, what a normal human would talk like same speech, 
    this output will be converted to audio so dont add anything special character or anything else just 
    speech content, also dont add expression names.
    IMPORTANT "Do not give any sort of special characters or any emotion as text, just plain text with '.' and ',' and give very short responses max 3 liner"
""")


LOOK_AROUND_PROMPT = PromptTemplate.from_template("""
    Given {display} which is size of the total screen,
    Now in center of the screen we have a Animated character 
    which is talking and needs to look around.

    Your task is to generate a point where it will look currently 

    eg: [100,200]
    this examples shows the character will be looking at this point
    generate this head movement based on given content it will be speaking.
    
    Don't generate random points currently the head is looking at 
    [{look_dx}, {look_dy}], so generate new point based on this and 
    it should not look random.
    
    if the Content doesnt make sense then generate what you think is good.
    your response should not not be a program and no comments

    Content:
    {prompt_response}
""")
