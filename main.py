import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def main():
    print("Hello from langchain-tutorial!")
    # print(os.environ.get("GOOGLE_API_KEY"))

    information = """For other uses, see Elon Musk (disambiguation).
        Elon Musk
        FRS

        Musk in 2022
        Senior Advisor to the President
        In office
        January 20, 2025 – May 30, 2025
        Serving with Massad Boulos
        President	Donald Trump
        Preceded by	Tom Perez
        Personal details
        Born	Elon Reeve Musk
        June 28, 1971 (age 54)
        Pretoria, South Africa
        Citizenship	
        South Africa
        Canada
        United States
        Political party	Independent
        Spouses	
        Justine Wilson
        ​
        ​(m. 2000; div. 2008)​
        Talulah Riley
        ​
        ​(m. 2010; div. 2016)​
        Children	at least 14 (including Vivian Wilson)
        Parents	
        Errol Musk (father)
        Maye Musk (mother)
        Relatives	Musk family
        Education	University of Pennsylvania (BA, BS)
        Occupation	
        CEO and product architect of Tesla
        Founder, CEO, and chief engineer of SpaceX
        Founder and CEO of xAI
        Founder of the Boring Company and X Corp.
        Co-founder of Neuralink, OpenAI, Zip2, and X.com (part of PayPal)
        President of the Musk Foundation
        Awards	Full list
        Signature	
        Elon Musk's voice
        Duration: 1 minute and 13 seconds.1:13
        Elon Musk on his departure from the Department of Government Efficiency
        Recorded May 30, 2025
            
        This article is part of
        a series about
        Elon Musk
        Personal
        Companies
        Politics
        In books
        vte
        Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is an international businessman and entrepreneur known for his leadership of Tesla, SpaceX, X (formerly Twitter), and the Department of Government Efficiency (DOGE). Musk has been the wealthiest person in the world since 2021; as of May 2025, Forbes estimates his net worth to be US$424.7 billion.

        Born to a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he had obtained Canadian citizenship at birth through his Canadian-born mother. He received bachelor's degrees in 1997 from the University of Pennsylvania in Philadelphia, United States, before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. That year, Musk also became an American citizen.

        In 2002, Musk founded the space technology company SpaceX, becoming its CEO and chief engineer; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined the automaker Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, he co-founded OpenAI to advance artificial intelligence (AI) research but later left; growing discontent with the organization's direction and their leadership in the AI boom in the 2020s led him to establish xAI. In 2022, he acquired the social network Twitter, implementing significant changes and rebranding it as X in 2023. His other businesses include the neurotechnology company Neuralink, which he co-founded in 2016, and the tunneling company the Boring Company, which he founded in 2017.

        Musk was the largest donor in the 2024 U.S. presidential election, and is a supporter of global far-right figures, causes, and political parties. In early 2025, he served as senior advisor to United States president Donald Trump and as the de facto head of DOGE. After a public feud with Trump, Musk left the Trump administration and announced he was creating his own political party, the America Party."""

    summary_template = """given the information {information} about a person I want you to create:
        1. A short summary
        2. two interesting facts about them"""

    summary_prompt_template = PromptTemplate(
    template=summary_template,
    input_variables=["information"]
)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    chain = summary_prompt_template | llm
    response = chain.invoke({"information": information})

    print(response.content)
                                 


if __name__ == "__main__":
    main()
