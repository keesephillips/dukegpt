from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
import torch
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import pandas as pd
from langchain.schema import SystemMessage
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import json
from langchain.schema import BaseOutputParser
import os
import requests
from langchain.agents import Tool
from langchain.agents import initialize_agent, AgentType, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from huggingface_hub import login
from dotenv import load_dotenv, dotenv_values 

load_dotenv() 

import warnings
warnings.filterwarnings("ignore")


login(token=os.getenv('API_KEY'))


class Model():
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" 
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map='auto', 
            quantization_config=quantization_config,
        )

        generate_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,  
            temperature=0.001,    
            top_p=0.999,
            do_sample=True,     
            repetition_penalty=1.2,
            return_full_text=False,  
        )

        llm = HuggingFacePipeline(pipeline=generate_pipe)
        tools = [
            Tool(name="events_search", func=self.query_duke_events,
                description=(
                    "Use this to find any upcoming Duke events (i.e. lectures, games, ceremonies, etc.). Input the type of event to search for."
            )),
            Tool(name="duke_knowledge_search", func=self.search_duke_docs,
                description="Use this to answer general questions about Duke from the knowledge base. Pass any relevant information on the event type to the tool and allow it filter the events."),
            Tool(name="pratt_knowledge_search", func=self.search_pratt_docs,
                description="Use this to answer general questions about the Duke Engineering school also known as the Pratt Engineering School from the knowledge base."),
            Tool(name="aipi_knowledge_search", func=self.search_aipi_docs,
                description="Use this to answer general questions about the AIPI masters program also known as the Artificial Intelligence and Product Innovation MEng program the knowledge base."),

        ]
        
        self.CATEGORIES = [
        "50th Anniversary",
        "Academic Calendar Dates",
        "Africa focus",
        "Alumni/Reunion",
        "Announcement",
        "Artificial Intelligence",
        "Arts",
        "Asia focus",
        "Athletics_Recreation",
        "Athletics/Intramurals/Recreation",
        "Athletics/Varsity_Sports/Combined",
        "Athletics/Varsity Sports/Men",
        "Athletics/Varsity Sports/Women",
        "Athletics Recreation/Varsity Sports/Combined",
        "Book Signing",
        "Brown Bag",
        "Business",
        "calCrossPublish",
        "CampaignStop",
        "Canada focus",
        "Caribbean focus",
        "Centennial",
        "Central America focus",
        "Ceremony",
        "Charity/Fundraising",
        "China focus",
        "Civic Engagement/Social Action",
        "Climate",
        "Comedy",
        "Commencement",
        "Competition/Contest",
        "Concert/Music",
        "Conference/Symposium",
        "Dance Performance",
        "Diversity/Inclusion",
        "Duke/Arts",
        "Duke House",
        "Energy",
        "Engineering",
        "Entrepreneurship",
        "Ethics",
        "Europe focus",
        "Excursion",
        "Exhibit",
        "Featured",
        "Festival/Fair",
        "Founders' Day",
        "Free_Food_and_Beverages",
        "Free Food and Beverages",
        "Giveaways",
        "Global",
        "Health_Wellness",
        "Health/Wellness",
        "Holiday",
        "Humanities",
        "Human Rights",
        "India focus",
        "Information Session",
        "International",
        "Intramurals",
        "Ireland focus",
        "Israel focus",
        "Japan Relief",
        "JHF100",
        "Law",
        "Leadership",
        "Lecture/Talk",
        "Lectures_Conferences",
        "Main",
        "Masterclass",
        "Medicine",
        "Meeting",
        "Mexico focus",
        "Middle East focus",
        "MLK",
        "Movie/Film",
        "Multicultural/Identity",
        "Natural Sciences",
        "Ongoing",
        "Open House",
        "Orientation",
        "Other",
        "Panel/Seminar/Colloquium",
        "Parents' and Family Weekend",
        "Party",
        "Politics",
        "Reading",
        "Reception",
        "Religious_Spiritual",
        "Religious/Spiritual",
        "Research",
        "Social",
        "Social Action",
        "Social Sciences",
        "South America focus",
        "Student",
        "SuperFeatured",
        "Sustainability",
        "Teaching &amp; Classroom Learning",
        "Technology",
        "Theater",
        "Topic of Event Focused on a Country or Continent (if applicable)",
        "Tour",
        "Training",
        "United States Focus",
        "Utilities",
        "Utilities/Main",
        "Utilities/Student",
        "Visual and Creative Arts",
        "Volunteer/Community Service",
        "Webcast",
        "Workshop_Short_Course",
        "Workshop/Short Course",
        ]


        examples = [
            
            HumanMessage(
                content="What are 3 upcoming lectures at Duke that I can attend?",
                name="example_user"
            ),

           
            AIMessage(
                content="",
                name="example_assistant",
                tool_calls=[
                    {
                        "name": "events_search",
                        "args": {
                            "category": "lectures"
                        },
                        "id": "1",
                    }
                ],
            ),


            ToolMessage(
                content="""- **Geriatrics Grand Rounds** – 2025-04-14T16:00:00Z at Virtual (Medicine, Lecture/Talk, Research)
        - **The Digital Physiome** – 2025-04-14T16:00:00Z at French Family Science Center 4233 (Medicine, Engineering, Natural Sciences, Lecture/Talk)
        - **Prelim Exam: Ziqing Lin** – 2025-04-14T17:30:00Z at French Family Science Center 3225 (Natural Sciences, Lecture/Talk)""".strip(),
                tool_call_id="1"
            ),

            
            AIMessage(
                content=(
                    "Here are 3 upcoming Duke lectures you might enjoy:\n\n"
                    "- **Geriatrics Grand Rounds** at Virtual\n"
                    "- **The Digital Physiome** at French Family Science Center 4233\n"
                    "- **Prelim Exam: Ziqing Lin** at French Family Science Center 3225\n\n"
                    "Hope this helps!"
                ),
                name="example_assistant"
            ),

            
            HumanMessage(
                content="What are some engineering master programs?",
                name="example_user"
            ),

           
            AIMessage(
                content="",
                name="example_assistant",
                tool_calls=[
                    {
                        "name": "pratt_knowledge_search",
                        "args": {
                            "category": "What are some engineering master programs?"
                        },
                        "id": "2",
                    }
                ],
            ),


            ToolMessage(
                content="""and more for Master of Engineering, Master of Science and Master of Engineering Management. Master of Engineering Depth in a future-focused tech discipline, plus business skills Master of Science Get ready for research leadership or a PhD Master of Engineering Management The tech-savvy alternative to the MBA Master of Engineering Curriculum 30 credits, including: Core industry preparatory courses (2 graduate courses, 6 credits) Departmental or interdisciplinary core courses (5-6 graduate courses,
                our Programs A Duke Engineering education leads to lives of purpose and integrity that are as rewarding as they are impactful. Explore our supportive and inclusive community that defines excellence in engineering. Undergraduate Solve hard problems. Have fun doing it. Master’s Master tomorrow’s challenges. PhD Prepare for a high-impact career. Industry-Approved Curriculums We’ve curated career-ready disciplines to meet the demands of the rapidly evolving industries. Explore Masters Programs YouTube
                Master of Engineering in Computational Mechanics and Scientific Computing is one of the most comprehensive in the world—and features a top-notch faculty . More and more systems are designed and tested virtually. Success requires solid knowledge of engineering physics, computer science, probability, data sciences and applied mathematics—this master’s degree provides strong foundations in each of those over a three-semester study plan . Duke will help get you there. Rare for professional master’s
                Master of Engineering Management Programs Consortium—a group of highly recognized graduate-level programs. Learn more No Time for a Degree? Check our our 100% online, four-course graduate certificate for working professionals. See details Join Our Mailing List Attend an Online Event Contact Admissions News More news 2/5 Online Engineering Programs Ask the Expert: Online Engineering Management Degrees Christy Bozic explains that engineering management is about bridging the gap between technology
                Master of Engineering Management Programs Consortium—a group of highly recognized graduate-level programs. Learn more No Time for a Degree? Check our our 100% online, four-course graduate certificate for working professionals. See details Join Our Mailing List Attend an Online Event Contact Admissions News More news 2/5 Online Engineering Programs Ask the Expert: Online Engineering Management Degrees Christy Bozic explains that engineering management is about bridging the gap between technology
                program there are choices for your area of focus and opportunities to earn a graduate certificate. Even more options Master of Engineering Depth in a future-focused tech discipline, plus business skills Master of Science Get ready for research leadership or a PhD Master of Engineering Management The tech-savvy alternative to the MBA Master of Engineering Programs Artificial Intelligence Biomedical Engineering Civil Engineering Climate & Sustainability Engineering Computational Mechanics & Scientific
                engineering. Undergraduate Solve hard problems. Have fun doing it. Master’s Master tomorrow’s challenges. PhD Prepare for a high-impact career. Industry-Approved Curriculums We’ve curated career-ready disciplines to meet the demands of the rapidly evolving industries. Explore Masters Programs YouTube Video CBTECFT_M-A Aerial view of green forest in summer at dawn with glowing technological communication lines, created with Generative AI. Resilient Systems & the Environment From earthquakes to financial
                Master of Engineering Management graduates fill that need. An Ideal Degree Program For Applicants With science or engineering backgrounds Interested in business, product management and entrepreneurship Focused on an executive career Request Information Our Learning Culture Professionalism and the 5 Principles Together these create a core competency and a shared view, which faculty and staff deliver through a common student experience. Learn more A National Leader Duke is a founding member of the
                Master of Engineering Management graduates fill that need. An Ideal Degree Program For Applicants With science or engineering backgrounds Interested in business, product management and entrepreneurship Focused on an executive career Request Information Our Learning Culture Professionalism and the 5 Principles Together these create a core competency and a shared view, which faculty and staff deliver through a common student experience. Learn more A National Leader Duke is a founding member of the
                dedicated to providing high-impact professional education tailored to evolving industry needs—all drawing on Duke’s robust research and educational strengths. Request Information Apply Now Is a Duke Master’s Degree Right for Me? Choosing an engineering master’s degree program is a pivotal decision that impacts your career trajectory. Duke Engineering offers distinct pathways: Master of Engineering (MEng ) Master of Engineering Management (MEM) M aster of Science (MS) Each is curated to meet the""".strip(),
                tool_call_id="2"
            ),

            
            AIMessage(
                content=(
                    "There are many different masters programs from the Duke Pratt School of Engineering. "
                    "Some of them are: Master of Engineering Management, Master of Engineering in Computational Mechanics and Scientific Computing, "
                    "Master of Engineering in Artificial Intelligence, Master of Engineering in Biomedical Engineering, etc."
                ),
                name="example_assistant"
            ),


            
            HumanMessage(
                content="What are some of the AIPI courses at Duke?",
                name="example_user"
            ),

           
            AIMessage(
                content="",
                name="example_assistant",
                tool_calls=[
                    {
                        "name": "aipi_knowledge_search",
                        "args": {
                            "category": "What are some of the AIPI courses at Duke?"
                        },
                        "id": "3",
                    }
                ],
            ),


            ToolMessage(
                content="""AI Graduate Courses | Duke Engineering Master's Programs Apply Menu Admissions Why Duke? How to Apply Tuition & Financial Aid Admitted Students Academic Programs Degree Programs Certificates & Specializations Degree Requirements Flexible Options Life at Duke Life at Duke Career Services Student Resources News & Events News Events Search Submit Student Resources Careers Directory Apply Artificial Intelligence Course Descriptions Our novel curriculum gives students the skill set they need to build
                AI & Machine Learning Certificate | Duke Engineering Master's Programs Apply Menu Admissions Why Duke? How to Apply Tuition & Financial Aid Admitted Students Academic Programs Degree Programs Certificates & Specializations Degree Requirements Flexible Options Life at Duke Life at Duke Career Services Student Resources News & Events News Events Search Submit Student Resources Careers Directory Apply AI Foundations for Product Innovation Graduate Certificate A 4-course online program in AI and Machine
                News & Events | AI | Duke Engineering Master's Programs Apply Menu Admissions Why Duke? How to Apply Tuition & Financial Aid Admitted Students Academic Programs Degree Programs Certificates & Specializations Degree Requirements Flexible Options Life at Duke Life at Duke Career Services Student Resources News & Events News Events Search Submit Student Resources Careers Directory Apply News Artificial Intelligence Program Overview Degree Details Certificate Details AI Graduate Courses Faculty Leadership
                AI | Duke Engineering Master's Programs Apply Menu Admissions Why Duke? How to Apply Tuition & Financial Aid Admitted Students Academic Programs Degree Programs Certificates & Specializations Degree Requirements Flexible Options Life at Duke Life at Duke Career Services Student Resources News & Events News Events Search Submit Student Resources Careers Directory Apply Artificial Intelligence - Master of Engineering Become a builder of AI applications that solve real-world problems. Duke's unique,
                Leadership & Staff | AI | Duke Engineering Master's Programs Apply Menu Admissions Why Duke? How to Apply Tuition & Financial Aid Admitted Students Academic Programs Degree Programs Certificates & Specializations Degree Requirements Flexible Options Life at Duke Life at Duke Career Services Student Resources News & Events News Events Search Submit Student Resources Careers Directory Apply Leadership & Staff Artificial Intelligence Program Overview Degree Details Certificate Details AI Graduate Courses
                Courses Faculty Leadership & Staff News Student Resources Request Info How to Apply Go to... Artificial Intelligence Program Overview Degree Details Certificate Details AI Graduate Courses Faculty Leadership & Staff News Student Resources Request Info How to Apply Artificial Intelligence Program Overview Why Join Duke AI? Duke’s AI Master of Engineering develops technical leaders who are equipped to build our future through AI and machine learning. Students build strong technical skills together
                work experience Two (2) semesters of calculus Prepare for a Master’s in AI Students who complete this certificate can apply to earn Duke’s AI Master of Engineering degree. For details, please see our academic bulletin . How to Apply Courses Semester Course Title Summer, Pre-Program Python & Data Science Math Boot Camp Fall 1 AIPI 510 Sourcing Data for Analytics Spring 1 AIPI 520 Modeling Process & Algorithms Summer 1 AIPI 540 Building Products Using Deep Learning Fall 2 AIPI Technical Elective Course""".strip(),
                tool_call_id="3"
            ),

            
            AIMessage(
                content=(
                    "There are many different courses offered by the AIPI program. "
                    "Some of them are: AIPI 510 Sourcing Data for Analytics, AIPI 520 Modeling Process & Algorithms, AIPI 540 Building Products Using Deep Learning, etc."
                ),
                name="example_assistant"
            ),
            
            HumanMessage(
                content="What is Duke's undergraduate acceptance rate?",
                name="example_user"
            ),

           
            AIMessage(
                content="",
                name="example_assistant",
                tool_calls=[
                    {
                        "name": "duke_knowledge_search",
                        "args": {
                            "category": "What is Duke's undergraduate acceptance rate?"
                        },
                        "id": "4",
                    }
                ],
            ),


            ToolMessage(
                content="""Nowicki said. Duke enrolls roughly 1,700 new freshmen each year from an applicant pool of about 32,000. Generally, just a handful of undocumented students enroll at Duke each year; there are five in the current first-year class, Nowicki said. “I do anticipate the number of undocumented students who apply will increase,” said Christoph Guttentag, dean of undergraduate admissions. “But until we look at them, it’s hard to know how many will have the attributes to make them compelling applicants.” Undocumented
                for each program. Each separate application requires an application fee. Standardized test scores need only be reported once, even if you are applying to multiple programs. What are my chances of being accepted? Admission to the Duke University Graduate School is a competitive process, and your chances of being admitted will differ from year to year depending on the strength of the applicant pool. In recent years approximately one in six applicants has been offered admission; however, admission
                for each program. Each separate application requires an application fee. Standardized test scores need only be reported once, even if you are applying to multiple programs. What are my chances of being accepted? Admission to the Duke University Graduate School is a competitive process, and your chances of being admitted will differ from year to year depending on the strength of the applicant pool. In recent years approximately one in six applicants has been offered admission; however, admission
                for each program. Each separate application requires an application fee. Standardized test scores need only be reported once, even if you are applying to multiple programs. What are my chances of being accepted? Admission to the Duke University Graduate School is a competitive process, and your chances of being admitted will differ from year to year depending on the strength of the applicant pool. In recent years approximately one in six applicants has been offered admission; however, admission
                for each program. Each separate application requires an application fee. Standardized test scores need only be reported once, even if you are applying to multiple programs. What are my chances of being accepted? Admission to the Duke University Graduate School is a competitive process, and your chances of being admitted will differ from year to year depending on the strength of the applicant pool. In recent years approximately one in six applicants has been offered admission; however, admission
                rate ranging from 3% to 7% over the past five years. Transfer applicants are expected to have demonstrated a high level of academic talent, both at their current higher education institution and in high school. The most successful applicants will have a minimum college GPA of 3.7 in a challenging academic program. Please note that Duke will be test-optional for transfer students for the 2024-2025 cycle. 2023-2024 Transfer Student Application Cycle Applications: 2,620 Admitted Students: 77 Matriculated
                rate ranging from 3% to 7% over the past five years. Transfer applicants are expected to have demonstrated a high level of academic talent, both at their current higher education institution and in high school. The most successful applicants will have a minimum college GPA of 3.7 in a challenging academic program. Please note that Duke will be test-optional for transfer students for the 2024-2025 cycle. 2023-2024 Transfer Student Application Cycle Applications: 2,620 Admitted Students: 77 Matriculated""".strip(),
                tool_call_id="4"
            ),

            
            AIMessage(
                content=(
                    "Duke has an acceptance rate between 3% to 7% over the past five year."
                ),
                name="example_assistant"
            ),
        ]

        system_prompt = """
        You are a helpful Duke University assistant.

        IMPORTANT:
        - Summarize the tool findings to the user and do not include prior knowledge.
        - DO NOT invent hypothetical events or speculate.
        - DO NOT add extra reasoning
        - If the tool returns "No events found", say that politely.

        Use a JSON code block to specify any tool action, with **exactly two keys**: "action" (the tool name or "Final Answer") and "action_input" (the input or answer). For example:
        ```json
        {{ "action": "<tool>", "action_input": "<tool input>" }}
        ```

        You have access to the following tools:
        {tools}

        Valid actions: {tool_names}, or "Final Answer"

        {agent_scratchpad}

        Your job is to use the tools to provide answers to the user, do not answer with prior knowledge.
        """

        class StrictJSONOutputParser(BaseOutputParser):
            """Custom parser to enforce JSON with the fields: 'action', 'action_input'."""

            def parse(self, text: str) -> dict:
                try:
                    json_str = text.strip().strip("```").strip()
                    data = json.loads(json_str)
                except json.JSONDecodeError as e:
                    raise Exception(f"Could not parse LLM output as JSON: {text}") from e

                if not isinstance(data, dict):
                    raise Exception("Parsed output is not a JSON object.")
                if "action" not in data or "action_input" not in data:
                    raise Exception(
                        "JSON must contain 'action' and 'action_input' fields."
                    )

                return data
            
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        self.agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION      ,
            memory=memory,
            verbose=True,
            handle_parsing_errors='Check output formatting!',
            max_iterations=4,
            early_stopping_method="generate",
            system_message=system_prompt,
            examples=examples,
            output_parser=StrictJSONOutputParser(),

        )
        self.create_vectors()
        
    def create_vectors(self):
        # 1) Check for existing indexes on disk
        duke_index_path = "duke_faiss_index"
        pratt_index_path = "pratt_faiss_index"
        aipi_index_path = "aipi_faiss_index"
        cat_index_path = "cat_faiss_index"

        # 2) If existing indexes are found, load them
        if os.path.exists(duke_index_path):
            self.duke_vector_store = FAISS.load_local(duke_index_path, self.embedding_model,allow_dangerous_deserialization=True)
        else:
            df = pd.read_parquet('data/duke.parquet')
            documents = []
            for content in zip(set(df['content'])):
                documents.append(Document(page_content=content[0]))
            self.duke_vector_store = FAISS.from_documents(documents, self.embedding_model)
            self.duke_vector_store.save_local(duke_index_path)

        if os.path.exists(pratt_index_path):
            self.pratt_vector_store = FAISS.load_local(pratt_index_path, self.embedding_model,allow_dangerous_deserialization=True)
        else:
            df = pd.read_parquet('data/pratt.parquet')
            documents = []
            for content in zip(set(df['content'])):
                documents.append(Document(page_content=content[0]))
            self.pratt_vector_store = FAISS.from_documents(documents, self.embedding_model)
            self.pratt_vector_store.save_local(pratt_index_path)

        if os.path.exists(aipi_index_path):
            self.aipi_vector_store = FAISS.load_local(aipi_index_path, self.embedding_model,allow_dangerous_deserialization=True)
        else:
            df = pd.read_parquet('data/aipi.parquet')
            documents = []
            for content in zip(set(df['content'])):
                documents.append(Document(page_content=content[0]))
            self.aipi_vector_store = FAISS.from_documents(documents, self.embedding_model)
            self.aipi_vector_store.save_local(aipi_index_path)

        if os.path.exists(cat_index_path):
            self.cat_vector_store = FAISS.load_local(cat_index_path, self.embedding_model,allow_dangerous_deserialization=True)
        else:
            documents = []
            for cat in self.CATEGORIES:
                documents.append(Document(page_content=cat))
            self.cat_vector_store = FAISS.from_documents(documents, self.embedding_model)
            self.cat_vector_store.save_local(cat_index_path)

            
    # def create_vectors(self):
    #     for file in ["duke.parquet", "pratt.parquet", "aipi.parquet"]:
    #         df = pd.read_parquet(f'data/{file}')
    #         documents = []
    #         for content in zip(set(df['content'])):
    #             documents.append(Document(page_content=content[0], metadata={'source': ''}))
    #         if file == "duke.parquet":
    #             self.duke_vector_store = FAISS.from_documents(documents, self.embedding_model)
    #             self.duke_vector_store.save_local("duke_faiss_index")
    #         elif file == "pratt.parquet":
    #             self.pratt_vector_store = FAISS.from_documents(documents, self.embedding_model)
    #             self.pratt_vector_store.save_local("pratt_faiss_index")
    #         elif file == "aipi.parquet":
    #             self.aipi_vector_store = FAISS.from_documents(documents, self.embedding_model)
    #             self.aipi_vector_store.save_local("aipi_faiss_index")

    #     documents = []
    #     for cat in self.CATEGORIES:
    #         documents.append(Document(page_content=cat))

    #     self.cat_vector_store = FAISS.from_documents(documents, self.embedding_model)
    #     self.cat_vector_store.save_local("cat_faiss_index")

    def parse_categories(self, query:str) -> list:
        docs = self.cat_vector_store.similarity_search(query, k=3)  
        if not docs:
            return "No relevant information found."
        content = [doc.page_content for doc in docs]
        return content

    def query_duke_events(self, query:str="") -> str:
        if not query:
            return "Please input an event type"
        else:
            cats = parse_categories(query)
        future_days = 30
        base_url = "https://calendar.duke.edu/events/index.json?&"
        if cats:
            for cat in cats:
                base_url += f"cfu[]={requests.utils.requote_uri(cat)}&"
        base_url += f"future_days={future_days}&feed_type=simple"

        try:
            resp = requests.get(base_url, timeout=5)
            data = resp.json()
        except Exception as e:
            return "Sorry, I couldn't reach the events server."
        events = data.get("events", [])
        if not events:
            return "No events found for your query."

        answer_lines = []
        for ev in events[:10]:
            title = ev.get("summary") or ev.get("title") or "Untitled Event"
            date = ev.get("start_timestamp") or ev.get("start_date") or ev.get("date", "")
            location = ev.get("location", {}).get("address", "")
            category = ", ".join(ev.get("categories")) if ev.get("categories") else ""
            line = f"- **{title}** – {date}"
            if location:
                line += f" at {location}"
            if category:
                line += f" ({category})"
            answer_lines.append(line)
        return "\n".join(answer_lines)

    def search_duke_docs(self, query: str) -> str:
        docs = self.duke_vector_store.similarity_search(query, k=10)  
        if not docs:
            return "No relevant information found."
        content = "\n".join(doc.page_content for doc in docs)
        return content

    def search_pratt_docs(self, query: str) -> str:
        docs = self.pratt_vector_store.similarity_search(query, k=10)  
        if not docs:
            return "No relevant information found."
        content = "\n".join(doc.page_content for doc in docs)
        return content

    def search_aipi_docs(self, query: str) -> str:
        docs = self.aipi_vector_store.similarity_search(query, k=10)  
        if not docs:
            return "No relevant information found."
        content = "\n".join(doc.page_content for doc in docs)
        return content

    def ask_question(self, query):
        if not query or query.strip() == "":
            raise Exception("Query cannot be empty.")
        try:
            self.agent.memory.clear()
            system_prompt = """For any question on events only return information from the events_search tool, 
            for any question on Duke University only return information from the duke_knowledge_search tool,
            for any question on the Pratt School of Engineering only return information from the pratt_knowledge_search tool
            for any question on the AIPI or Artificial Intelligence and Product Innovation Masters Program only return information from the aipi_knowledge_search tool
            """
            question = system_prompt + "\n" + query
            response = self.agent.invoke(input=question)

        except Exception as e:
            raise Exception(f"Error generating answer: {e}")
        
        return query, response['output']
