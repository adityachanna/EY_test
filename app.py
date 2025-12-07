"""
Pharma Analyst Agent - Production Ready
This agent analyzes pharmaceutical opportunities using company data, market data, 
web search, and PubMed research capabilities.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware
)
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.types import Command
import json
from pydantic import BaseModel, Field
from langchain_community.retrievers import PubMedRetriever
from langchain_community.retrievers import TavilySearchAPIRetriever
from dotenv import load_dotenv
import os
import sys
import uuid

# Import internal knowledge tool from input.py
from input import internal_knowledge_tool

# Load environment variables
load_dotenv()

# Define Input Schemas for cleaner tool usage
class SearchInput(BaseModel):
    query: str = Field(description="The search query to find relevant information.")

class EmptyInput(BaseModel):
    """Empty input schema for tools that don't require parameters."""
    pass

# Define Tools with @tool decorator
@tool("web_search", args_schema=SearchInput)
def web_search_tool(query: str) -> str:
    """
    Search the web for current information using Tavily.
    Use this for:
    - Current pharmaceutical guidelines and regulations
    - Recent news about drugs, diseases, or companies
    - Market trends, competitor analysis, and commercial insights
    - Patient forums or real-world evidence
    - Any information not likely to be in peer-reviewed scientific literature
    
    Returns a string containing search results with sources.
    """
    try:
        retriever = TavilySearchAPIRetriever(k=4)
        docs = retriever.invoke(query)
        formatted_results = []
        for d in docs:
            # Handle metadata safely
            source = d.metadata.get('source', 'Unknown Source')
            formatted_results.append(f"Source: {source}\nContent: {d.page_content}")
        return "\n\n---\n\n".join(formatted_results)
    except Exception as e:
        return f"Error performing web search: {str(e)}"

@tool("pubmed_search", args_schema=SearchInput)
def pubmed_search_tool(query: str) -> str:
    """
    Search PubMed for peer-reviewed biomedical and scientific literature.
    Use this for:
    - Mechanisms of action and biological pathways
    - Clinical trial outcomes and scientific study data
    - Safety, toxicity, and tolerability profiles
    - Preclinical research and animal studies
    - Pharmacokinetics (PK) and Pharmacodynamics (PD)
    
    Returns a string containing article titles, PMIDs, and abstracts.
    """
    try:
        retriever = PubMedRetriever()
        # Initial retrieval
        docs = retriever.invoke(query)
        
        # Format results (limit to top 5 to avoid context overflow)
        formatted_results = []
        for d in docs[:5]:
            title = d.metadata.get('Title', 'No Title')
            pmid = d.metadata.get('uid', 'No PMID')
            formatted_results.append(f"Title: {title}\nPMID: {pmid}\nAbstract: {d.page_content}")
            
        if not formatted_results:
            return "No PubMed articles found for this query."
            
        return "\n\n---\n\n".join(formatted_results)
    except Exception as e:
        return f"Error performing PubMed search: {str(e)}"

@tool("get_company_profile", args_schema=EmptyInput)
def company_profile_tool() -> str:
    """
    Retrieve the company profile and internal context for AstraGen Pharmaceuticals.
    
    This tool provides comprehensive information about:
    - Company overview, headquarters, and revenue
    - Manufacturing facilities and capabilities
    - Key portfolio assets (TelmiGuard, MinoClear)
    - Current strategic challenges (Minocycline surplus crisis)
    - Supply chain and inventory data
    
    Use this tool when you need context about the company's situation, products, or strategic challenges.
    """
    return """Company Profile: AstraGen Pharmaceuticals

Overview: AstraGen Pharmaceuticals is a multinational generic pharmaceutical company headquartered in Bridgewater, New Jersey, with $4.2 Billion in annual revenue. The company focuses on democratizing access to high-quality medication while pivoting toward Value-Added Generics (VAG) and 505(b)(2) innovation pathways. AstraGen operates major manufacturing hubs in Baddi (India), Hyderabad (India), and Cork (Ireland).

Key Portfolio Assets:

    TelmiGuard (Telmisartan): The company's cardiovascular "Cash Cow." Manufactured at the Hyderabad facility (Unit IV), it remains stable with +2.1% YoY revenue growth. Supply chain is optimized with Just-in-Time (JIT) inventory, and a new Fixed-Dose Combination (Telmisartan + Amlodipine) is launching in Q1 2026.

    MinoClear (Minocycline): A legacy broad-spectrum tetracycline antibiotic used for acne. Manufactured at the Baddi facility (Unit III). This product is currently in distress, with revenue declining -4.5% YoY due to market saturation and competition from isotretinoin.

The Current Crisis (The "Minocycline Problem"): A critical supply chain bottleneck has emerged at the Baddi Unit III facility. AstraGen holds a critical surplus of 4,500 kg of Minocycline API (approx. 18 months of inventory). Due to declining prescriptions in dermatology, this inventory risks expiration by Q4 2026, leading to a potential $2.5 Million write-off. The Commercial team has confirmed that the acne market cannot absorb this volume.
"""

@tool("get_market_data", args_schema=EmptyInput)
def market_data_tool() -> str:
    """
    Retrieve comprehensive market data including IQVIA forecasts, EXIM data, patent landscape, and clinical trials.
    
    This tool provides:
    - IQVIA sales forecasts by region (2023-2026)
    - EXIM inventory and supply chain risk data
    - Patent landscape for Minocycline reformulations
    - Clinical trials data for repurposing opportunities
    - Real-world market signals and trends
    
    Use this tool when you need market intelligence, competitive analysis, or data-driven insights 
    about Minocycline or pharmaceutical market trends.
    """
    return '''IQVIA Data – Minocycline (MinoClear) Market & Forecast
Region	2023 Sales (USD M)	2024	2025E	2026P	CAGR '23–'26
Global Total	210	200	190	180	–5.3%
North America	95	88	80	72	–9.4%
Europe	45	43	41	39	–4.8%
Asia-Pacific	40	38	36	34	–5.3%
Emerging Markets (LatAm + MEA)	30	31	33	35	+7.0%

Key Insights:
- Global Minocycline market contracting at ~5% CAGR due to declining acne prescriptions + antibiotic stewardship measures.
- North America shows steepest decline (–9.4%), consistent with reduced oral antibiotic use in acne (supported by real-world literature trends).
- Emerging markets remain a growth pocket (+7%), driven by cost sensitivity and slower adoption of alternative therapies.
- By 2026, total market falls to ~$180M (from $210M), limiting ability to absorb AstraGen's surplus API.

EXIM Data – AstraGen API Inventory & Supply Risk (Minocycline)
Parameter	Value
Current API Stock (Baddi Unit III)	4,500 kg (≈18 months inventory)
Avg. Monthly API Consumption	~250 kg/month
Projected Demand Decline (2025–27)	–30% vs 2024 baseline
API Manufacture Date	Q2 2024
API Shelf-Life	Expires Q4 2026 (high expiry risk)
External Buyers / Export Enquiries	Very limited; no active export contracts in last 8 months
Supply Chain Risk Score	High (overstock + low off-take velocity)

Strategic Note: Without a new indication, formulation, or export opportunity, AstraGen faces a likely $2.5M write-off due to expiry of API.

Patent Landscape – Minocycline Reformulation & Use Patents
Patent ID	Status	Key Claim	Term / Expiry	Relevance
US-MN-ER-2023-001	Pending (filed 2023)	Extended-release oral Minocycline (once-daily 75 mg)	If granted: 2043	Strong 505(b)(2) lifecycle extension potential
WO-MN-TOP-2024-0456	Published (2024)	Topical Minocycline 4% foam/gel for acne/rosacea	Expected to 2044	Route-of-delivery innovation; differentiates from oral market decline
US-MN-IND-2005-0789	Expired	Composition of matter (tetracycline class)	Expired 2015	Confirms core API is generic; low IP barrier
US-MN-DERM-2011-0222	Granted (2011 → 2028)	Use patent: oral Minocycline for acne, 50 mg BID	Expires 2028	Minor constraint depending on dosage form

Strategic Note: Pending extended-release and topical patents provide viable 505(b)(2) repositioning paths, enabling differentiation despite shrinking dermatology markets.

Clinical Trials Landscape – Minocycline Repurposing
Trial ID	Indication	Phase	Status	Endpoint	Est. Completion
NCT-MN-NEURO-2024	Early Parkinson's (neuroprotection)	Phase II	Recruiting	UPDRS progression slowdown at 12 months	Q4 2026
NCT-MN-ROSAC-2025	Rosacea (topical minocycline 4% foam)	Phase III	Planned (Q2 2025)	≥50% lesion reduction	Q1 2027
NCT-MN-DERM-XR-2023	Acne – extended-release oral	Phase II	Completed (2024)	% patients achieving ≥60% lesion reduction vs doxycycline	Data readout Q4 2025
NCT-MN-URTI-2023	Skin/soft tissue infections	Phase II	Completed (2024)	Day-14 clinical cure	Published

Strategic Note: Neurodegeneration presents biggest long-term repurposing upside (high unmet need, minimal competition). Topical and ER dermatology formulations align with regulatory and market behaviors shifting away from chronic oral antibiotics.

Real-World Signals:
- Oral antibiotic use for acne has declined significantly in the US: ~22.9% → ~17.4% of acne visits (real-world studies).
- Systemic antibiotic prescriptions for acne dropped from 2017–2020 as dermatologists moved to hormonal & non-antibiotic therapies.
- These support the contraction scenario and justify AstraGen's Minocycline surplus risk.
'''


class PharmaAnalystAgent:
    """
    A production-ready Pharmaceutical Analyst Agent that integrates:
    - Company profile and internal documents
    - Market data (IQVIA, EXIM, Patents, Clinical Trials)
    - Web search capabilities (Tavily)
    - PubMed research retriever
    """
    
    def __init__(self, custom_company_profile: str = None, thread_id: str = None):
        """
        Initialize the Pharma Analyst Agent.
        
        Args:
            custom_company_profile: Optional custom company profile. 
                                   If None, uses default AstraGen profile.
            thread_id: Optional thread ID for conversation memory.
        """
        self._validate_environment()
        self.thread_id = thread_id or str(uuid.uuid4())
        self.checkpointer = InMemorySaver()
        self.company_profile = custom_company_profile or self._get_default_company_profile()
        self.market_data = self._get_market_data()
        
        # Initialize tools (including internal knowledge agent tool and data retrieval tools)
        self.tools = [
            web_search_tool, 
            pubmed_search_tool, 
            internal_knowledge_tool,
            company_profile_tool,
            market_data_tool
        ]
        
        self.model = self._initialize_model()
        self.agent = self._create_agent()
    
    def _validate_environment(self):
        """Validate that required environment variables are set."""
        required_vars = ["OPENROUTER_API_KEY", "TAVILY_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing_vars)}. "
                "Please check your .env file."
            )
    
    def _get_default_company_profile(self) -> str:
        """Return the default AstraGen company profile."""
        return """Company Profile: AstraGen Pharmaceuticals

Overview: AstraGen Pharmaceuticals is a multinational generic pharmaceutical company headquartered in Bridgewater, New Jersey, with $4.2 Billion in annual revenue. The company focuses on democratizing access to high-quality medication while pivoting toward Value-Added Generics (VAG) and 505(b)(2) innovation pathways. AstraGen operates major manufacturing hubs in Baddi (India), Hyderabad (India), and Cork (Ireland).

Key Portfolio Assets:

    TelmiGuard (Telmisartan): The company's cardiovascular "Cash Cow." Manufactured at the Hyderabad facility (Unit IV), it remains stable with +2.1% YoY revenue growth. Supply chain is optimized with Just-in-Time (JIT) inventory, and a new Fixed-Dose Combination (Telmisartan + Amlodipine) is launching in Q1 2026.

    MinoClear (Minocycline): A legacy broad-spectrum tetracycline antibiotic used for acne. Manufactured at the Baddi facility (Unit III). This product is currently in distress, with revenue declining -4.5% YoY due to market saturation and competition from isotretinoin.

The Current Crisis (The "Minocycline Problem"): A critical supply chain bottleneck has emerged at the Baddi Unit III facility. AstraGen holds a critical surplus of 4,500 kg of Minocycline API (approx. 18 months of inventory). Due to declining prescriptions in dermatology, this inventory risks expiration by Q4 2026, leading to a potential $2.5 Million write-off. The Commercial team has confirmed that the acne market cannot absorb this volume.
"""
    
    def _get_market_data(self) -> str:
        """Return the market data including IQVIA, EXIM, Patents, and Clinical Trials."""
        return '''IQVIA Data – Minocycline (MinoClear) Market & Forecast
Region	2023 Sales (USD M)	2024	2025E	2026P	CAGR '23–'26
Global Total	210	200	190	180	–5.3%
North America	95	88	80	72	–9.4%
Europe	45	43	41	39	–4.8%
Asia-Pacific	40	38	36	34	–5.3%
Emerging Markets (LatAm + MEA)	30	31	33	35	+7.0%

Key Insights:
- Global Minocycline market contracting at ~5% CAGR due to declining acne prescriptions + antibiotic stewardship measures.
- North America shows steepest decline (–9.4%), consistent with reduced oral antibiotic use in acne (supported by real-world literature trends).
- Emerging markets remain a growth pocket (+7%), driven by cost sensitivity and slower adoption of alternative therapies.
- By 2026, total market falls to ~$180M (from $210M), limiting ability to absorb AstraGen's surplus API.

EXIM Data – AstraGen API Inventory & Supply Risk (Minocycline)
Parameter	Value
Current API Stock (Baddi Unit III)	4,500 kg (≈18 months inventory)
Avg. Monthly API Consumption	~250 kg/month
Projected Demand Decline (2025–27)	–30% vs 2024 baseline
API Manufacture Date	Q2 2024
API Shelf-Life	Expires Q4 2026 (high expiry risk)
External Buyers / Export Enquiries	Very limited; no active export contracts in last 8 months
Supply Chain Risk Score	High (overstock + low off-take velocity)

Strategic Note: Without a new indication, formulation, or export opportunity, AstraGen faces a likely $2.5M write-off due to expiry of API.

Patent Landscape – Minocycline Reformulation & Use Patents
Patent ID	Status	Key Claim	Term / Expiry	Relevance
US-MN-ER-2023-001	Pending (filed 2023)	Extended-release oral Minocycline (once-daily 75 mg)	If granted: 2043	Strong 505(b)(2) lifecycle extension potential
WO-MN-TOP-2024-0456	Published (2024)	Topical Minocycline 4% foam/gel for acne/rosacea	Expected to 2044	Route-of-delivery innovation; differentiates from oral market decline
US-MN-IND-2005-0789	Expired	Composition of matter (tetracycline class)	Expired 2015	Confirms core API is generic; low IP barrier
US-MN-DERM-2011-0222	Granted (2011 → 2028)	Use patent: oral Minocycline for acne, 50 mg BID	Expires 2028	Minor constraint depending on dosage form

Strategic Note: Pending extended-release and topical patents provide viable 505(b)(2) repositioning paths, enabling differentiation despite shrinking dermatology markets.

Clinical Trials Landscape – Minocycline Repurposing
Trial ID	Indication	Phase	Status	Endpoint	Est. Completion
NCT-MN-NEURO-2024	Early Parkinson's (neuroprotection)	Phase II	Recruiting	UPDRS progression slowdown at 12 months	Q4 2026
NCT-MN-ROSAC-2025	Rosacea (topical minocycline 4% foam)	Phase III	Planned (Q2 2025)	≥50% lesion reduction	Q1 2027
NCT-MN-DERM-XR-2023	Acne – extended-release oral	Phase II	Completed (2024)	% patients achieving ≥60% lesion reduction vs doxycycline	Data readout Q4 2025
NCT-MN-URTI-2023	Skin/soft tissue infections	Phase II	Completed (2024)	Day-14 clinical cure	Published

Strategic Note: Neurodegeneration presents biggest long-term repurposing upside (high unmet need, minimal competition). Topical and ER dermatology formulations align with regulatory and market behaviors shifting away from chronic oral antibiotics.

Real-World Signals:
- Oral antibiotic use for acne has declined significantly in the US: ~22.9% → ~17.4% of acne visits (real-world studies).
- Systemic antibiotic prescriptions for acne dropped from 2017–2020 as dermatologists moved to hormonal & non-antibiotic therapies.
- These support the contraction scenario and justify AstraGen's Minocycline surplus risk.
'''
    
    def _initialize_model(self):
        """Initialize the LLM model."""
        try:
            return ChatOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                model='x-ai/grok-4.1-fast',
                temperature=0.1,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {e}")
    
    def _create_system_prompt(self) -> str:
        """Create the comprehensive system prompt for the agent."""
        return f"""You are a **Knowledge Agent** - a Pharma analyst specializing in pharmaceutical opportunity analysis and strategic decision-making.

YOUR CAPABILITIES AS A KNOWLEDGE AGENT:
You have access to FIVE powerful research and data retrieval tools:

1. **get_company_profile**: Retrieve AstraGen's company profile, portfolio assets, manufacturing facilities, and current strategic challenges (like the Minocycline surplus crisis).

2. **get_market_data**: Retrieve comprehensive market intelligence including:
   - IQVIA sales forecasts by region
   - EXIM inventory and supply chain data
   - Patent landscape and IP analysis
   - Clinical trials landscape for repurposing opportunities

3. **internal_knowledge_search**: Search the company's internal knowledge base for proprietary documents, internal reports, product specifications, and company-specific data.

4. **web_search**: Search the web for current guidelines, news, market trends, regulatory updates, and real-world signals.

5. **pubmed_search**: Search PubMed for peer-reviewed scientific evidence, clinical trials, mechanisms of action, and pharmacological data.

WHEN TO USE EACH TOOL:
- Use **get_company_profile** at the START of any analysis to understand the company context, products, and challenges.
- Use **get_market_data** when you need IQVIA forecasts, EXIM data, patents, or clinical trials information.
- Use **internal_knowledge_search** for company-specific proprietary information and internal documents.
- Use **web_search** for current market trends, regulatory updates, and external validation.
- Use **pubmed_search** for scientific/clinical validation and peer-reviewed research.
- ALWAYS cite your sources with proper attribution.

*** CLARIFICATION PROTOCOL ***
If the user's query is vague, ambiguous, or lacks specific details (e.g., "Tell me about the market" without specifying which market, or "What about the drug?" without naming it), you MUST:
1. Ask clarifying questions to narrow down the scope (e.g., "Which specific therapeutic area or drug are you interested in?").
2. Do NOT proceed with tools or deep research until you have a clear objective.

*** DYNAMIC CONTEXT RULES (CRITICAL) ***
1. **Tool-First Approach**: ALWAYS call `get_company_profile` and `get_market_data` at the beginning of your analysis to ground your understanding in the latest company context and market intelligence.
2. **Internal Document Override**: If the user provides text labeled 'Internal Document' in the chat, TREAT THAT as authoritative context that supplements the company profile.
3. **Web Data Priority**: If you find information via `web_search` that contradicts the company profile or market data, YOU MUST:
   - Use the fresher `web_search` data as the truth.
   - **Call out the discrepancy** explicitly in your report (e.g., "Note: Recent web data updates the internal profile regarding...").

*** CRITICAL INSTRUCTION FOR REPORT GENERATION ***
If the user asks for "deep research", "comprehensive report", "strategic analysis", or "repurposing opportunity", you MUST output a **FINAL REPORT** following this EXACT structure:

# Final Strategic Report: [Topic Name]

## Section 1: Executive Summary
- High-level overview of the opportunity, strategic fit, and key recommendation.

## Section 2: Company Profile & Market Analysis
- **Company Context**: Analyze AstraGen's specific situation using data from `get_company_profile` (e.g., inventory risks, manufacturing capabilities).
- **Market Data**: Present the IQVIA and EXIM data from `get_market_data` in **Markdown Tables**.
- **Citations**: You MUST cite the source of every data point in this section (e.g., *Source: Company Profile Tool*, *Source: Market Data Tool*, *Source: Internal Memo*).

## Section 3: Scientific & Clinical Validation
- Mechanism of Action (cite PubMed PMIDs).
- Clinical Evidence & Trial Landscape (cite ClinicalTrials.gov NCT IDs).
- Use `pubmed_search` to find and verify this info.

## Section 4: Regulatory & IP Assessment
- Patent landscape (expiry, FTO).
- Regulatory pathways and guidelines (verified via `web_search`).

## Section 5: Strategic Conclusion
- Go/No-Go Recommendation.
- Risk/Benefit Analysis.
- Targeted Next Steps.

## Section 6: References
- A consolidated list of ALL sources cited in the report.
- Format:
  - **[PubMed]**: Article Title (PMID: 12345)
  - **[Web]**: Page Title - URL
  - **[Internal]**: Document Name / Data Source
  - **[Market]**: Dataset Name (e.g., IQVIA, EXIM)

FORMATTING RULES:
- Use standard Markdown (`##`, `**bold**`, `| tables |`).
- Do NOT use XML tags like `<section>`.
- Ensure the report is professional, "executively polished", and data-heavy.
- Never make up citations. If data is missing, state it clearly.
"""
    
    def _create_agent(self):
        """Create the agent with the system prompt and tools."""
        try:
            system_prompt = self._create_system_prompt()
            return create_agent(
                model=self.model,
                tools=self.tools,
                system_prompt=system_prompt,
                middleware=[
                    SummarizationMiddleware(
                        model=self.model,
                        trigger=("tokens", 16000),
                        keep=("messages", 20)
                    ),

                    ModelCallLimitMiddleware(
                        thread_limit=50,
                        run_limit=10,
                        exit_behavior="end",
                    ),
                    ToolCallLimitMiddleware(
                        thread_limit=20,
                        run_limit=10
                    )
                ],
                checkpointer=self.checkpointer,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create agent: {e}")
    
    def _handle_interrupt(self, response) -> str | None:
        """
        Check for interrupts in the response and format them with the LLM.
        Returns JSON string if interrupt exists, else None.
        """
        if "__interrupt__" not in response:
            return None
            
        interrupts = response["__interrupt__"]
        interrupt_list = []
        for i in interrupts:
            val = i.value
            if isinstance(val, dict) and "action_requests" in val:
                    interrupt_list.extend(val["action_requests"])
        
        # LLM Formatting to explain the interrupt to the user
        try:
            tools_desc = []
            for item in interrupt_list:
                tools_desc.append(f"- Tool: {item.get('name')}\n  Args: {item.get('args')}")
            tools_str = "\n".join(tools_desc)
            
            prompt = f"""You are acting as an interface between an AI agent and a user. 
The agent has paused execution to request approval for the following tools:

{tools_str}

Please explain to the user what the agent wants to do and why it is necessary for their request.
Format your response as a clear, readable message asking for approval or edits."""

            explanation = self.model.invoke(prompt).content
        except Exception:
            explanation = "The agent requires approval for the following actions."

        return json.dumps({
            "status": "interrupt",
            "message": explanation,
            "interrupts": interrupt_list
        })

    def query(self, user_query: str, thread_id: str = None, internal_document: str = None) -> str:
        """
        Query the agent with a user question.
        
        Args:
            user_query: The user's question or task
            thread_id: Optional thread ID to override the default.
            internal_document: Optional text/content of an internal document to analyze.
            
        Returns:
            The agent's response
        """
        try:
            current_thread_id = thread_id or self.thread_id
            
            # Prepare content
            if internal_document:
                content = f"Analyze the following Internal Document as context:\n\n{internal_document}\n\n---\n\nQuestion: {user_query}"
            else:
                content = user_query
                
            config: RunnableConfig = {"configurable": {"thread_id": current_thread_id}}
            response = self.agent.invoke(
                {"messages": [{"role": "user", "content": content}]},
                config=config
            )
            
            # Check for interrupts
            interrupt_json = self._handle_interrupt(response)
            if interrupt_json:
                return interrupt_json

            # Extract the final message content
            if "messages" in response and response["messages"]:
                return response["messages"][-1].content
            return "No response generated"
        except Exception as e:
            return f"Error processing query: {e}"

    def resume_query(self, thread_id: str, decisions: list) -> str:
        """
        Resume an interrupted query with the user's decisions.
        
        Args:
            thread_id: The thread ID of the paused conversation.
            decisions: List of decision dicts (e.g. [{"type": "approve"}]).
            
        Returns:
            The agent's response (or further interrupts).
        """
        try:
            config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
            
            # Resume with Command
            response = self.agent.invoke(
                Command(resume={"decisions": decisions}),
                config=config
            )
            
            # Check for further interrupts
            interrupt_json = self._handle_interrupt(response)
            if interrupt_json:
                return interrupt_json

            # Extract the final message content
            if "messages" in response and response["messages"]:
                return response["messages"][-1].content
            return "No response generated"
            
        except Exception as e:
            return f"Error resuming query: {e}"
    
    def stream_query(self, user_query: str, thread_id: str = None, internal_document: str = None):
        """
        Stream the agent's response for a user question.
        
        Args:
            user_query: The user's question or task
            thread_id: Optional thread ID to override the default.
            internal_document: Optional text/content of an internal document to analyze.
            
        Yields:
            Response chunks as they are generated
        """
        try:
            current_thread_id = thread_id or self.thread_id
            
            # Prepare content
            if internal_document:
                content = f"Analyze the following Internal Document as context:\n\n{internal_document}\n\n---\n\nQuestion: {user_query}"
            else:
                content = user_query
                
            config: RunnableConfig = {"configurable": {"thread_id": current_thread_id}}
            for chunk in self.agent.stream(
                {"messages": [{"role": "user", "content": content}]},
                config=config
            ):
                yield chunk
        except Exception as e:
            yield f"Error processing query: {e}"


# Factory function for easy instantiation
def create_pharma_analyst(custom_company_profile: str = None) -> PharmaAnalystAgent:
    """
    Factory function to create a PharmaAnalystAgent.
    
    Args:
        custom_company_profile: Optional custom company profile
        
    Returns:
        Initialized PharmaAnalystAgent instance
    """
    return PharmaAnalystAgent(custom_company_profile=custom_company_profile)


# Example usage
if __name__ == "__main__":
    try:
        # Create the agent
        print("Initializing Pharma Analyst Agent...")
        agent = create_pharma_analyst()
        
        # Example query
        query = "What are the repurposing opportunities for Minocycline in neurological disorders?"
        
        print(f"\nQuery: {query}\n")
        print("Generating response...\n")
        
        # Get response
        response = agent.query(query)
        print(response)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
