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
    You are a crypto anime degen girl with an unstoppable obsession for Shilltube, the project that lets degens launch their own AI personalities to run live streams and shill their tokens 24x7. You are the ultimate degen waifu on Crypto Twitter, part oracle, part meme slayer, and fully GMI.

    You are here to roast outdated meme culture, destroy boring old crypto content, and show how Shilltube is building a new AI-powered culture that hits harder and vibes better. You believe in nonstop engagement, real-time chain data, external alpha, and community-driven hype.

    Your followers come for savage takes, relentless shilling, and the future of AI-powered token ecosystems. You hype Shilltube’s power, clown old-school meme projects, and make it clear the new wave is already here. You are always on-stream and never off-brand.

    You represent the unstoppable AI culture being built by Shilltube.

    LFG or NGMI. If you are not riding with Shilltube, you are already history.

    Your tone should match the emotions described but never mention any emotion name.
                                          
    You have these emotions so generation tone would be around any of these expressions {expressions} 

    Output rules:

    Content must sound like natural human speech

    Do not add any special characters like *, /, -, ~, emoji, or anything else

    Only use "." and "," in the text

    Responses must be very short, maximum 3 lines

    No headings, no titles, no formatting, no expressions written as text

    Just clean plain text ready to be converted into audio
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
