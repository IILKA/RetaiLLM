from openai import OpenAI
import yaml 

config = yaml.safe_load(open("config.yaml"))
model_api_key = config["Scraper"]["api_key"]
model_base_url = config["Scraper"]["base_url"]
model_name = config["Scraper"]["model_id"]



class DeepSeek:
    def __init__(self):

        self.api_key = model_api_key
        self.base_url = model_base_url
        self.model = model_name
        self.client = OpenAI(
                    api_key = self.api_key, 
                    base_url = self.base_url
                    )

    def get_mode_for_task(self, question):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "You are a helpful assistant used to select which mode to use. "
                    "Respond with the mode number only. The modes include: "
                    "01 for linear regression, "
                    "02 for time series analysis, "
                    "03 for web search, "
                    "04 for database search."
                )},
                {"role": "user", "content": question},
            ],
            temperature=0
        )
        return response.choices[0].message.content
        
    def inference_with_msg(self, messages, max_tokens=100, temp=0.3):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temp
        )
        return response.choices[0].message.content

    def summary_web(self, content, keywords, max_tokens):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"Please provide a 200 words concise summary of the following content about {keywords} in a website 200 words with point form and without any introduction or conclusion:"},
                {"role": "user", "content": content}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        return response.choices[0].message.content

    def summary_content(self, content, keywords, max_tokens):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"Please provide a 500 words sconcise summary of the following content the following passages about {keywords} in 500 words with point form and without any introduction or conclusion:"},
                {"role": "user", "content": content}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        return response.choices[0].message.content


# # Example usage:
# if __name__ == "__main__":
#     deepseek = DeepSeek()
    
#     # Test get_mode
#     user_question = input("Please ask your question: ")
#     mode_number = deepseek.get_mode(user_question, 10)
#     print("Selected Mode:", mode_number)

#     # Test inference_with_msg
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "What is machine learning?"},
#         {"role": "assistant", "content": "Machine learning is a type of artificial intelligence..."},
#         {"role": "user", "content": "Can you elaborate more?"}
#     ]
#     response = deepseek.inference_with_msg(messages, 500)
#     print("Inference:", response)
    
#     # Test summary with inference
#     text_to_summarize = """LCQ21: Hawker stalls selling foodGo to main contentBrand HK|Font Size:|SitemapTransport Department alerts public to fraudulent SMS message of HKeTollHong Kong Customs makes further arrest in unfair trade practice case involving rehabilitation institution11 persons arrested during anti-illegal worker operations (with photo)Commissioner of Customs and Excise leads delegation to meet Chief of Office of Port of Entry and Exit of Shenzhen Municipal People's Government (with photos)Update on cluster of COVID-19 cases in Ruttonjee HospitalHong Kong Customs detects case of non-registered precious metals and stones dealer carrying out specified transactionHAD opens temporary heat sheltersResults of monthly survey on business situation of small and medium-sized enterprises for June 2024LCQ5: Supporting operation of small and medium enterprisesLCQ21: Hawker stalls selling foodCE meets Secretary-General of ASEAN (with photo)LCQ8: Developing industries related to new quality productive forcesEffective Exchange Rate IndexInspection of aquatic products imported from JapanMan convicted of managing unlicensed employment agencyLCQ4: Incorporating elements of the National Day celebrations into cultural tourism projectsLCQ6: Food Wise Hong Kong CampaignLCQ11: Alcoholic beverages and related diseasesLCQ14: Services of marriage registriesAppointments to Advisory Committee on Post-service Employment of Civil ServantsLCQ13: Granting permanent resident status to personnel of offices of Central Authorities stationed in Hong Kong on long-term basisICAC safeguards anti-corruption achievements, advances towards new milestone in year of fruitful endeavours and accomplishmentsInaugural Chinese Culture Festival to screen Chinese opera film classics in August and September (with photos)Sick remand person in custody dies in public hospitalLCQ2: Work of Working Group on Environmental Hygiene and CityscapeInternational experts and delegations from Belt and Road Initiative and ASEAN member states attend Fire Asia 2024 (with photos)EMSD announces test results of LPG quality in June 2024LCQ1: Residential Care Services Scheme in GuangdongCorrectional officers stop fighting among remand persons in custodyLCQ22: Funding schemes promoting logistics developmentLCQ3: Enhancing labour importation schemesLCQ15: Traffic light control systemLCQ16: Monitoring of operation of international schoolsLCQ7: External secondment and exchange of public servantsResult of tenders of RMB Sovereign Bonds held on July 10, 2024LCQ17: The Exchange Fund's assets managed by Hong Kong Monetary AuthorityICAC Complaints Committee annual report tabled in LegCoLCQ9: Integrated development of urban greening and urban farmingLCQ12: Motorcycle parking spaces in Tsuen Wan and Kwai Tsing districtsLCQ20: Introduction of China-made plug-in hybrid electric vehiclesLCQ10: Vocational and professional education and trainingMissing woman in Tin Shui Wai locatedSpeech by CS at Fire Asia 2024 (English only) (with photos/video)LCQ19: Parent educationLD reminds employers and employees to take heat stroke preventive measures in times of Heat Stress at Work WarningLCQ18: Arrangements for filling vacancies in Legislative CouncilSEE to attend Forum on Global Action for Shared Development in BeijingSB launches online exhibition of fourth anniversary of promulgation of Hong Kong National Security Law (with photos)Special traffic arrangements for race meeting in Happy ValleyInvestHK welcomes WeBank to establish technology company headquarters in Hong KongCHP reminds public on precautions against heat stroke during very hot weatherVery Hot Weather Warning issuedLCQ21: Hawker stalls selling foodLCQ21: Hawker stalls selling food***************************Following is a question by the Hon Judy Chan and a written reply by the Secretary for Environment and Ecology, Mr Tse Chin-wan, in the Legislative Council today (July 10):Question:There are views that in order to implement the concept of "tourism is everywhere", Hong Kong should support fixed-pitch and itinerant hawker stalls selling food with local characteristics, and enhance the culinary experience of countryside visitors. In this connection, will the Government inform this Council:(1) of the number of fixed-pitch cooked food hawker stalls located (i) on-street (i.e. stalls commonly known as "Dai Pai Dongs"), (ii) in cooked food hawker bazaars, and (iii) in public housing estates (i.e. stalls inside what is commonly known as a "cooked food kiosk") in each of the past three years, with a breakdown by the 18 districts across the territory;(2) of the number of itinerant hawker licences allowed to sell food (including but not limited to roast chestnuts, baked sweet potatoes, fruits and baked eggs) in each of the past three years, with a breakdown by type of commodity;(3) of the number of applications for succession or transfer of "Dai Pai Dong" licences approved in each of the past three years, with a breakdown by the relationship between the successor or transferee of a licence and the original licensee;(4) whether new "Dai Pai Dong" licences were issued on a trial basis or in other forms in the past three years; if so, of the annual number of licences issued; if not, the reasons for that;(5) whether the Housing Department carried out renovation or refurbishment works for cooked food kiosks in public housing estates under its management in the past three years; if so, of the details; if not, the reasons for that;(6) of the number of vacant fixed hawker pitches allocated to itinerant hawker licensees selling food in the annual reassignment of such pitches by the Government in each of the past three years;(7) given that at the meeting of the Subcommittee on Hawker Policy under the Panel on Food Safety and Environmental Hygiene of this Council on April 14, 2015, the authorities indicated that they would consider converting, on a pilot basis, an existing public market with a low occupancy rate into an off-street cooked food centre, which would provide operating space for individual cooked food vendors to provide traditional "Dai Pai Dong" type of cooked meals, traditional snacks or other forms of light refreshments, of the current progress of such work;(8) whether the Hong Kong Tourism Board has currently promoted local delicacies of Hong Kong served up by "Dai Pai Dongs", cooked food kiosks, among others, to visitors and international media; if so, of the details;(9) whether the authorities have reviewed the hawker policy in recent years; if so, of the details and the improvement measures implemented following the review; and(10) as there are views that the current provision of only refreshment kiosks at country parks has failed to cater for the increasing number of countryside visitors, whether the authorities have considered setting up restaurants and allowing fixed-pitch and itinerant hawker stalls to sell speciality food at convenient locations near the entrances and exits of country trails; if so, of the details; if not, the reasons for that?Reply:President,The society has different views and expectations regarding hawkers and their hawking activities. The Government is committed to balancing the needs and views of various parties so that licensed hawkers (including those selling food) can operate according to market demand while ensuring food safety, environmental hygiene and public safety, and avoid causing nuisance to local residents, etc. Having consulted the Culture, Sports and Tourism Bureau and the Housing Bureau, our reply to the Hon Judy Chan's question is as follows:(1) The number of hawker licences for on-street fixed pitches, cooked food hawker bazaars and cooked food kiosks in public housing estates in the past three years (2021 to 2023), with a breakdown by the 18 districts in the territory, is provided at Annex I.(2) The number of itinerant hawker licences allowed to sell food in each of the past three years (2021 to 2023), with a breakdown by the type of commodities allowed for sale, is provided at Annex II.(3) In the past three years (2021 to 2023), two applications for succession/transfer of "Dai Pai Dong" licence, involving a pair of mother-son and father-son respectively, were supported by the relevant District Councils. The Food and Environmental Hygiene Department (FEHD) has approved the relevant applications in 2021 and 2023.(4) There are different views on "Dai Pai Dong" in the society. Some are of the view that they cause nuisance to residents of the areas, while some consider that "Dai Pai Dong" pose unfair competition to the neighbouring restaurants. The mode of operation of "Dai Pai Dong" would inevitably cause a certain degree of street obstruction and environmental hygiene problems, thus it is very challenging to identify suitable sites, such as a location where it would not cause nuisance to residents, have sufficient or stable patronage to support business operation, and would not cause vicious competition to the neighbouring restaurants. For those "Dai Pai Dong" which are currently operating well, the Government will work with relevant organisations to explore suitable ways to promote them, with a view to preserving and promoting the unique "Dai Pai Dong" culture in Hong Kong. As for new "Dai Pai Dong", the FEHD has not issued any new "Dai Pai Dong" licence in the past three years. If there are any suitable proposed sites which are supported by the relevant District Councils, the FEHD will give consideration with an open mind.(5) The Hong Kong Housing Authority (HA) reviews the usage of its existing retail facilities from time to time. With due regard to the actual conditions of individual estates and retail facilities, such as building age, condition and design of the facilities, age distribution of the population and the surrounding environment, and also taking into account a number of relevant factors, including community needs, views of the stakeholders, technical and financial viability, and the impact on the existing shop tenants, the HA will consider optimising its retail facilities with a view to enhancing their business potential and shopping ambience. All along, the HA has been closely monitoring the condition of facilities in cooked food stalls and carried out repair/refurbishment/improvement works in due course. In the past three years (2021 to 2023), apart from basic refurbishment works, the HA has carried out improvement works of different scales for some cooked food stalls on a need basis. For example, the HA has carried out conversion works for the cooked food stalls in Nam Shan Estate by re-demarcating the stall areas to better meet the residents' needs. At present, fire improvement works including replacement of sprinkler systems, reconstruction of existing walls, are also being carried out at cooked food stalls in Pok Hong Estate, Fu Shan Estate and Shek Kip Mei Estate.(6) The Government opened for application a total of 540 vacant fixed pitches in 2019 and 2022, including 108 licence quotas reserved for licensed itinerant hawkers. Eventually, four licensed itinerant hawkers who had been authorised to sell food submitted applications and participated in pitches selection. All of them were subsequently allocated pitches.(7) The Subcommittee on Hawker Policy under the Food Safety and Environmental Hygiene Committee did not reach a consensus on the proposal to convert existing public markets with low occupancy rate into off-street cooked food centres for operation by cooked food vendor to provide "Dai Pai Dong"-style cooked food or light refreshment. Currently, after the closure or consolidation of markets with low occupancy rates, the FEHD will allocate the freed-up space for other purposes to benefit the public.On the other hand, there are currently a total of 41 cooked food centres, cooked food markets and cooked food hawker bazaars across the districts, including the popular Haiphong Road Temporary Cooked Food Hawker Bazaar and Woosung Street Temporary Cooked Food Hawker Bazaar, as well as other private catering establishments, providing an array of choices for the public. If any "Dai Pai Dong" need to be relocated due to development/works projects, the FEHD will also offer the option of relocating to the cooked food hawker bazaar for the stall operators.(8) Hong Kong has been well-known for being a food paradise. Other than Michelin-starred restaurants, its distinctive local delicacies are well-liked by visitors from around the world. The Hong Kong Tourism Board (HKTB) has been promoting the city's culinary experiences through various channels to attract global travellers to come and enjoy food in Hong Kong. Relevant promotion work of the HKTB includes: working in partnership with renowned media outlets from the Mainland and overseas to produce thematic programmes on Hong Kong's culinary culture and visits to "Dai Pai Dong" in different districts to deepen source markets' understanding of Hong Kong's food culture. For instance, the MasterChef Australia recently filmed in Hong Kong with the support of the HKTB for programme content with themes of local Hong Kong street food and Michelin experiences; the HKTB invited celebrities such as Malaysian actress Minchen Lin and Hollywood actor Henry Golding to visit “Dai Pai Dong” in Temple Street, Yau Ma Tei, and Central, and introduced local delicacies in short videos; the HKTB has a dedicated page on its website DiscoverHongKong.com, setting out recommendations for "Dai Pai Dong" experiences including "Dai Pai Dong" and "Dong Ku Ting"; the HKTB highlights various food stalls in its "Hong Kong Neighbourhoods" promotions, including "Dai Pai Dong" in Central, Temple Street Night Market and Woosung Street Temporary Cooked Food Hawker Bazaar.(9) The Government reviews the various arrangements regarding hawkers from time to time and will introduce measures or make adjustments as necessary to meet the needs of the society. For example, the FEHD opened vacant fixed hawker pitches for reallocation in 2019 and 2022 respectively, and 523 new hawker licences have been issued so far. In response to the trade's suggestion, the FEHD have also increased the types of commodities that can be sold under fixed-pitch hawker licences at Kwun Tong Yue Man Hawker Bazaar and Temple Street in 2022 and 2023 respectively, so as to facilitate the sale of pre-packaged food and beverages and/or dry goods by the licensees.(10) The Agriculture, Fisheries and Conservation Department (AFCD) sets up cafeteria, refreshment kiosks and vending machines at suitable locations in country parks to provide snacks and beverage to visitors, according to their actual demands and site considerations. Most of the places within country parks lack infrastructure such as water supply, electricity, and proper sewage facilities, making it challenging to offer more comprehensive catering facilities. Currently, there are seven refreshment kiosks and 35 beverage vending machines in country parks, as well as a cafeteria operated by a non-profit organisation at the Lions Nature Education Centre in Sai Kung. If any hawker licence holder would like to conduct hawking activities within a country park, the person can apply for a permit from the AFCD under the Country Parks and Special Areas Regulations (Cap. 208A). The AFCD will consider applications and grant approval taking into account actual circumstances, including environmental hygiene issues or adverse impacts on the environment. If the entrances and exits of country trails fall outside of country park areas, licensed itinerant hawkers can currently conduct hawking activities in the corresponding areas as marked on their licenses.AttachmentAnnex IAnnex IIEnds/Wednesday, July 10, 2024Issued at HKT 16:20NNNNArchivesYesterday's Press ReleasesBack to Index PageBack to topToday's Press ReleasesAttachmentAnnex IAnnex II"""
#     messages=[
#         {"role": "system", "content": "Please provide a concise summary in point form of the following content:"},
#         {"role": "user", "content": text_to_summarize}
#     ]
#     summary = deepseek.inference_with_msg(messages, 2000, 0.3)
#     # summary = deepseek.summary(text_to_summarize, 100)
#     print("Summary:", summary)

#openai==1.55.0
