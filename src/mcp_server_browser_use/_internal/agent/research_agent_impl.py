import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, TypedDict, Optional, Sequence
import threading

# Langchain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool, StructuredTool
from langchain_community.tools.file_management import WriteFileTool, ReadFileTool, ListDirectoryTool
from pydantic import BaseModel, Field

from browser_use.browser.browser import BrowserConfig
from browser_use.browser.context import BrowserContextWindowSize

# Langgraph imports
from langgraph.graph import StateGraph
from ..controller.custom_controller import CustomController
from ..browser.custom_browser import CustomBrowser
from ..browser.custom_context import CustomBrowserContextConfig
from .task_agent_impl import BrowserUseAgent
from ..utils.mcp_client import setup_mcp_client_and_tools

logger = logging.getLogger(__name__)

# Constants
REPORT_FILENAME = "report.md"
PLAN_FILENAME = "research_plan.md"
SEARCH_INFO_FILENAME = "search_info.json"

_AGENT_STOP_FLAGS = {}
_BROWSER_AGENT_INSTANCES = {}


async def run_single_browser_task(
		task_query: str,
		task_id: str,
		llm: Any,  # Pass the main LLM
		browser_config: Dict[str, Any],
		stop_event: threading.Event,
		use_vision: bool = False,
) -> Dict[str, Any]:
	"""
	Runs a single BrowserUseAgent task.
	Manages browser creation and closing for this specific task.
	"""
	if not BrowserUseAgent:
		return {"query": task_query, "error": "BrowserUseAgent components not available."}

	# --- Browser Setup ---
	# These should ideally come from the main agent's config
	headless = browser_config.get("headless", False)
	window_w = browser_config.get("window_width", 1280)
	window_h = browser_config.get("window_height", 1100)
	browser_user_data_dir = browser_config.get("user_data_dir", None)
	use_own_browser = browser_config.get("use_own_browser", False)
	browser_binary_path = browser_config.get("browser_binary_path", None)
	wss_url = browser_config.get("wss_url", None)
	cdp_url = browser_config.get("cdp_url", None)
	disable_security = browser_config.get("disable_security", False)
	save_downloads_path = browser_config.get("save_downloads_path", None)
	trace_path = browser_config.get("trace_path", None)

	bu_browser = None
	bu_browser_context = None
	task_key = None
	try:
		logger.info(f"Starting browser task for query: {task_query}")
		extra_args = [f"--window-size={window_w},{window_h}"]
		if browser_user_data_dir:
			extra_args.append(f"--user-data-dir={browser_user_data_dir}")
		if use_own_browser:
			browser_binary_path = os.getenv("CHROME_PATH", None) or browser_binary_path
			if browser_binary_path == "":
				browser_binary_path = None
			chrome_user_data = os.getenv("CHROME_USER_DATA", None)
			if chrome_user_data:
				extra_args += [f"--user-data-dir={chrome_user_data}"]
		else:
			browser_binary_path = None

		bu_browser = CustomBrowser(
			config=BrowserConfig(
				headless=headless,
				disable_security=disable_security,
				browser_binary_path=browser_binary_path,
				extra_browser_args=extra_args,
				wss_url=wss_url,
				cdp_url=cdp_url,
			)
		)

		context_config = CustomBrowserContextConfig(
			save_downloads_path=save_downloads_path,
			trace_path=trace_path,
			browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h),
			force_new_context=True
		)
		bu_browser_context = await bu_browser.new_context(config=context_config)

		# Simple controller example, replace with your actual implementation if needed
		bu_controller = CustomController()

		# Construct the task prompt for BrowserUseAgent
		bu_task_prompt = f"""
		Browsing rule: When a search results page shows text like 'Showing results for' and also offers 'Search instead for <literal>', click the 'Search instead for' (or equivalent) to force the exact query. This applies to Brave, Bing, DuckDuckGo, and Google. Also look for similar UI like 'Did you mean' or 'Including results for' and prefer exact-match links.

		Research Task: {task_query}
			Objective: Find relevant information answering the query.
			Output Requirements: For each relevant piece of information found, please provide:
			1. A concise summary of the information.
			2. The title of the source page or document.
			3. The URL of the source.
			Focus on accuracy and relevance. Avoid irrelevant details.
			PDF cannot directly extract _content, please try to download first, then using read_file, if you can't save or read, please try other methods.
			"""

		bu_agent_instance = BrowserUseAgent(
			task=bu_task_prompt,
			llm=llm,  # Use the passed LLM
			browser=bu_browser,
			browser_context=bu_browser_context,
			controller=bu_controller,
			use_vision=use_vision,
		)

		# Store instance for potential stop() call
		task_key = f"{task_id}_{uuid.uuid4()}"
		_BROWSER_AGENT_INSTANCES[task_key] = bu_agent_instance

		if stop_event.is_set():
			logger.info(f"Browser task for '{task_query}' cancelled before start.")
			return {"query": task_query, "result": None, "status": "cancelled"}

		logger.info(f"Running BrowserUseAgent for: {task_query}")
		result = await bu_agent_instance.run()  # Assuming run is the main method
		logger.info(f"BrowserUseAgent finished for: {task_query}")

		final_data = result.final_result()

		if stop_event.is_set():
			logger.info(f"Browser task for '{task_query}' stopped during execution.")
			return {"query": task_query, "result": final_data, "status": "stopped"}
		else:
			logger.info(f"Browser result for '{task_query}': {final_data}")
			return {"query": task_query, "result": final_data, "status": "completed"}

	except Exception as e:
		logger.error(f"Error during browser task for query '{task_query}': {e}", exc_info=True)
		return {"query": task_query, "error": str(e), "status": "failed"}
	finally:
		if bu_browser_context:
			try:
				await bu_browser_context.close()
				bu_browser_context = None
				logger.info("Closed browser context.")
			except Exception as e:
				logger.error(f"Error closing browser context: {e}")
		if bu_browser:
			try:
				await bu_browser._close_without_httpxclients()
				bu_browser = None
				logger.info("Closed browser.")
			except Exception as e:
				logger.error(f"Error closing browser: {e}")

		if task_key and task_key in _BROWSER_AGENT_INSTANCES:
			del _BROWSER_AGENT_INSTANCES[task_key]


class BrowserSearchInput(BaseModel):
	queries: List[str] = Field(
		description=f"List of distinct search queries to find information relevant to the research task.")


async def _run_browser_search_tool(
	queries: List[str],
	task_id: str,  # Injected dependency
	llm: Any,  # Injected dependency
	browser_config: Dict[str, Any],
	stop_event: threading.Event,
	max_tabs: int = 1,
) -> List[Dict[str, Any]]:
	"""
	Execute browser searches based on LLM-provided queries.
	Opens a single browser window (context) and runs up to `max_tabs` concurrent agents (tabs) within it.
	"""

	if not queries:
		logger.info(f"[Browser Tool {task_id}] No queries provided, nothing to run.")
		return []

	logger.info(
		f"[Browser Tool {task_id}] Running {len(queries)} queries with up to {max_tabs} concurrent tabs in one window."
	)

	# Build an indexed queue to preserve input order in results
	queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
	for idx, q in enumerate(queries):
		queue.put_nowait((idx, q))

	results: list[Optional[Dict[str, Any]]] = [None] * len(queries)

	# Single browser + context to host multiple tabs
	bu_browser = None
	bu_browser_context = None
	controller = None
	try:
		headless = browser_config.get("headless", False)
		window_w = browser_config.get("window_width", 1280)
		window_h = browser_config.get("window_height", 1100)
		browser_user_data_dir = browser_config.get("user_data_dir", None)
		use_own_browser = browser_config.get("use_own_browser", False)
		browser_binary_path = browser_config.get("browser_binary_path", None)
		wss_url = browser_config.get("wss_url", None)
		cdp_url = browser_config.get("cdp_url", None)
		disable_security = browser_config.get("disable_security", False)
		save_downloads_path = browser_config.get("save_downloads_path", None)
		trace_path = browser_config.get("trace_path", None)

		extra_args = [f"--window-size={window_w},{window_h}"]
		if browser_user_data_dir:
			extra_args.append(f"--user-data-dir={browser_user_data_dir}")
		if use_own_browser:
			browser_binary_path = os.getenv("CHROME_PATH", None) or browser_binary_path
			if browser_binary_path == "":
				browser_binary_path = None
			chrome_user_data = os.getenv("CHROME_USER_DATA", None)
			if chrome_user_data:
				extra_args += [f"--user-data-dir={chrome_user_data}"]
		else:
			browser_binary_path = None

		bu_browser = CustomBrowser(
			config=BrowserConfig(
				headless=headless,
				disable_security=disable_security,
				browser_binary_path=browser_binary_path,
				extra_browser_args=extra_args,
				wss_url=wss_url,
				cdp_url=cdp_url,
			)
		)

		context_config = CustomBrowserContextConfig(
			save_downloads_path=save_downloads_path,
			trace_path=trace_path,
			browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h),
			force_new_context=False,
		)
		bu_browser_context = await bu_browser.new_context(config=context_config)
		controller = CustomController()

		sem = asyncio.Semaphore(max_tabs)

		async def run_one(idx: int, query: str):
			async with sem:
				if stop_event.is_set():
					results[idx] = {"query": query, "result": None, "status": "cancelled"}
					return
				bu_task_prompt = f"""
				Browsing rule: When a search results page shows text like 'Showing results for' and also offers 'Search instead for <literal>', click the 'Search instead for' (or equivalent) to force the exact query. This applies to Brave, Bing, DuckDuckGo, and Google. Also look for similar UI like 'Did you mean' or 'Including results for' and prefer exact-match links.

				Research Task: {query}
					Objective: Find relevant information answering the query.
					Output Requirements: For each relevant piece of information found, please provide:
					1. A concise summary of the information.
					2. The title of the source page or document.
					3. The URL of the source.
					Focus on accuracy and relevance. Avoid irrelevant details.
					PDF cannot directly extract _content, please try to download first, then using read_file, if you can't save or read, please try other methods.
				"""

				agent = BrowserUseAgent(
					task=bu_task_prompt,
					llm=llm,
					browser=bu_browser,
					browser_context=bu_browser_context,
					controller=controller,
					use_vision=False,
				)
				try:
					res = await agent.run()
					final_data = res.final_result()
					results[idx] = {"query": query, "result": final_data, "status": "completed"}
				except Exception as e:
					logger.error(f"[Browser Tool {task_id}] Error for query '{query}': {e}", exc_info=True)
					results[idx] = {"query": query, "error": str(e), "status": "failed"}

		tasks = [asyncio.create_task(run_one(i, q)) for i, q in enumerate(queries)]
		await asyncio.gather(*tasks)
	finally:
		try:
			if bu_browser_context:
				await bu_browser_context.close()
		except Exception:
			pass
		try:
			if bu_browser:
				await bu_browser._close_without_httpxclients()
		except Exception:
			pass

	processed_results: List[Dict[str, Any]] = []
	for i, q in enumerate(queries):
		item = results[i]
		if item is None:
			processed_results.append({"query": q, "error": "No result produced", "status": "failed"})
		else:
			processed_results.append(item)

	logger.info(f"[Browser Tool {task_id}] Finished search. Results count: {len(processed_results)}")
	return processed_results


def create_browser_search_tool(
		llm: Any,
		browser_config: Dict[str, Any],
		task_id: str,
		stop_event: threading.Event,
		max_tabs: int = 1,
) -> StructuredTool:
	"""Factory function to create the browser search tool with necessary dependencies."""
	from functools import partial
	bound_tool_func = partial(
		_run_browser_search_tool,
		task_id=task_id,
		llm=llm,
		browser_config=browser_config,
		stop_event=stop_event,
		max_tabs=max_tabs
	)

	return StructuredTool.from_function(
		coroutine=bound_tool_func,
		name="parallel_browser_search",
		description=(
			f"Use this tool to actively search the web for information related to a specific research task or question. "
			f"Executes searches with up to {max_tabs} concurrent tabs in a single browser window (context). "
			f"Provide a list of distinct search queries that are likely to yield relevant information."
		),
		args_schema=BrowserSearchInput,
	)


# --- Langgraph State Definition ---

class ResearchPlanItem(TypedDict):
	step: int
	task: str
	status: str  # "pending", "completed", "failed"
	queries: Optional[List[str]]  # Queries generated for this task
	result_summary: Optional[str]  # Optional brief summary after execution


class DeepResearchState(TypedDict):
	task_id: str
	topic: str
	research_plan: List[ResearchPlanItem]
	search_results: List[Dict[str, Any]]
	llm: Any
	tools: List[Tool]
	output_dir: Optional[Path]
	browser_config: Dict[str, Any]
	final_report: Optional[str]
	current_step_index: int
	stop_requested: bool
	error_message: Optional[str]
	messages: List[BaseMessage]


# --- Langgraph Nodes ---

def _load_previous_state(task_id: str, output_dir: str) -> Dict[str, Any]:
	"""Loads state from files if they exist."""
	state_updates = {}
	plan_file = os.path.join(output_dir, PLAN_FILENAME)
	search_file = os.path.join(output_dir, SEARCH_INFO_FILENAME)
	if os.path.exists(plan_file):
		try:
			with open(plan_file, 'r', encoding='utf-8') as f:
				plan = []
				step = 1
				for line in f:
					line = line.strip()
					if line.startswith(("- [x]", "- [ ]")):
						status = "completed" if line.startswith("- [x]") else "pending"
						task = line[5:].strip()
						plan.append(ResearchPlanItem(step=step, task=task, status=status, queries=None, result_summary=None))
						step += 1
				state_updates['research_plan'] = plan
				next_step = next((i for i, item in enumerate(plan) if item['status'] == 'pending'), len(plan))
				state_updates['current_step_index'] = next_step
				logger.info(f"Loaded research plan from {plan_file}, next step index: {next_step}")
		except Exception as e:
			logger.error(f"Failed to load or parse research plan {plan_file}: {e}")
			state_updates['error_message'] = f"Failed to load research plan: {e}"
	if os.path.exists(search_file):
		try:
			with open(search_file, 'r', encoding='utf-8') as f:
				state_updates['search_results'] = json.load(f)
				logger.info(f"Loaded search results from {search_file}")
		except Exception as e:
			logger.error(f"Failed to load search results {search_file}: {e}")
			state_updates['error_message'] = f"Failed to load search results: {e}"

	return state_updates


def _save_plan_to_md(plan: List[ResearchPlanItem], output_dir: str):
	"""Saves the research plan to a markdown checklist file."""
	plan_file = os.path.join(output_dir, PLAN_FILENAME)
	try:
		with open(plan_file, 'w', encoding='utf-8') as f:
			f.write("# Research Plan\n\n")
			for item in plan:
				marker = "- [x]" if item['status'] == 'completed' else "- [ ]"
				f.write(f"{marker} {item['task']}\n")
		logger.info(f"Research plan saved to {plan_file}")
	except Exception as e:
		logger.error(f"Failed to save research plan to {plan_file}: {e}")


def _save_search_results_to_json(results: List[Dict[str, Any]], output_dir: str):
	"""Overwrites search results to a JSON file."""
	search_file = os.path.join(output_dir, SEARCH_INFO_FILENAME)
	try:
		with open(search_file, 'w', encoding='utf-8') as f:
			json.dump(results, f, indent=2, ensure_ascii=False)
		logger.info(f"Search results saved to {search_file}")
	except Exception as e:
		logger.error(f"Failed to save search results to {search_file}: {e}")


def _save_report_to_md(report: str, output_dir: Path):
	"""Saves the final report to a markdown file."""
	report_file = os.path.join(output_dir, REPORT_FILENAME)
	try:
		with open(report_file, 'w', encoding='utf-8') as f:
			f.write(report)
		logger.info(f"Final report saved to {report_file}")
	except Exception as e:
		logger.error(f"Failed to save final report to {report_file}: {e}")


async def planning_node(state: DeepResearchState) -> Dict[str, Any]:
	"""Generates the initial research plan or refines it if resuming."""
	logger.info("--- Entering Planning Node ---")
	if state.get('stop_requested'):
		logger.info("Stop requested, skipping planning.")
		return {"stop_requested": True}

	llm = state['llm']
	topic = state['topic']
	existing_plan = state.get('research_plan')
	output_dir = state['output_dir']

	if existing_plan and state.get('current_step_index', 0) > 0:
		logger.info("Resuming with existing plan.")
		if output_dir:
			_save_plan_to_md(existing_plan, str(output_dir))
		return {"research_plan": existing_plan}

	logger.info(f"Generating new research plan for topic: {topic}")

	prompt = ChatPromptTemplate.from_messages([
		("system", """You are a meticulous research assistant. Your goal is to create a step-by-step research plan to thoroughly investigate a given topic.
		The plan should consist of clear, actionable research tasks or questions. Each step should logically build towards a comprehensive understanding.
		Format the output as a numbered list. Each item should represent a distinct research step or question.
		Example:
		1. Define the core concepts and terminology related to [Topic].
		2. Identify the key historical developments of [Topic].
		3. Analyze the current state-of-the-art and recent advancements in [Topic].
		4. Investigate the major challenges and limitations associated with [Topic].
		5. Explore the future trends and potential applications of [Topic].
		6. Summarize the findings and draw conclusions.

		Keep the plan focused and manageable. Aim for 5-10 detailed steps.
		"""),
		("human", f"Generate a research plan for the topic: {topic}")
	])

	try:
		response = await llm.ainvoke(prompt.format_prompt(topic=topic).to_messages())
		plan_text = response.content

		new_plan: List[ResearchPlanItem] = []
		for i, line in enumerate(plan_text.strip().split('\n')):
			line = line.strip()
			if line and (line[0].isdigit() or line.startswith(("*", "-"))):
				task_text = line.split('.', 1)[-1].strip() if line[0].isdigit() else line[1:].strip()
				if task_text:
					new_plan.append(ResearchPlanItem(
						step=i + 1,
						task=task_text,
						status="pending",
						queries=None,
						result_summary=None
					))

		if not new_plan:
			logger.error("LLM failed to generate a valid plan structure.")
			return {"error_message": "Failed to generate research plan structure."}

		logger.info(f"Generated research plan with {len(new_plan)} steps.")
		if output_dir:
			_save_plan_to_md(new_plan, str(output_dir))

		return {
			"research_plan": new_plan,
			"current_step_index": 0,
			"search_results": [],
		}

	except Exception as e:
		logger.error(f"Error during planning: {e}", exc_info=True)
		return {"error_message": f"LLM Error during planning: {e}"}


async def research_execution_node(state: DeepResearchState) -> Dict[str, Any]:
	"""
	Executes the next step in the research plan by invoking the LLM with tools.
	"""
	logger.info("--- Entering Research Execution Node ---")
	if state.get('stop_requested'):
		logger.info("Stop requested, skipping research execution.")
		return {"stop_requested": True, "current_step_index": state['current_step_index']}

	plan = state['research_plan']
	current_index = state['current_step_index']
	llm = state['llm']
	tools = state['tools']
	output_dir = str(state['output_dir']) if state['output_dir'] else None
	task_id = state['task_id']

	if not plan or current_index >= len(plan):
		logger.info("Research plan complete or empty.")
		return {}

	current_step = plan[current_index]
	if current_step['status'] == 'completed':
		logger.info(f"Step {current_step['step']} already completed, skipping.")
		return {"current_step_index": current_index + 1}

	logger.info(f"Executing research step {current_step['step']}: {current_step['task']}")

	llm_with_tools = llm.bind_tools(tools)
	if state['messages']:
		current_task_message = [HumanMessage(content=f"Research Task (Step {current_step['step']}): {current_step['task']}")]
		invocation_messages = state['messages'] + current_task_message
	else:
		current_task_message = [
			SystemMessage(content=(
				"You are a research assistant executing one step of a research plan. Use the available tools, "
				"especially the 'parallel_browser_search' tool, to gather information needed for the current task. "
				"Be precise with your search queries if using the browser tool."
			)),
			HumanMessage(content=f"Research Task (Step {current_step['step']}): {current_step['task']}")
		]
		invocation_messages = current_task_message

	try:
		logger.info(f"Invoking LLM with tools for task: {current_step['task']}")
		ai_response: BaseMessage = await llm_with_tools.ainvoke(invocation_messages)
		logger.info("LLM invocation complete.")

		tool_results = []
		executed_tool_names = []
		current_search_results = state.get('search_results', [])

		if not isinstance(ai_response, AIMessage) or not ai_response.tool_calls:
			logger.warning(
				f"LLM did not call any tool for step {current_step['step']}. Response: {getattr(ai_response, 'content', '')[:100]}..."
			)
			current_step['status'] = 'failed'
			current_step['result_summary'] = "LLM did not use a tool as expected."
			if output_dir:
				_save_plan_to_md(plan, output_dir)
			return {
				"research_plan": plan,
				"current_step_index": current_index + 1,
				"error_message": f"LLM failed to call a tool for step {current_step['step']}."
			}

		for tool_call in ai_response.tool_calls:
			tool_name = tool_call.get("name")
			tool_args = tool_call.get("args", {})
			tool_call_id = tool_call.get("id")

			logger.info(f"LLM requested tool call: {tool_name} with args: {tool_args}")
			executed_tool_names.append(tool_name)

			selected_tool = next((t for t in tools if t.name == tool_name), None)

			if not selected_tool:
				logger.error(f"LLM called tool '{tool_name}' which is not available.")
				tool_results.append(ToolMessage(
					content=f"Error: Tool '{tool_name}' not found.",
					tool_call_id=tool_call_id
				))
				continue

			try:
				stop_event = _AGENT_STOP_FLAGS.get(task_id)
				if stop_event and stop_event.is_set():
					logger.info(f"Stop requested before executing tool: {tool_name}")
					current_step['status'] = 'pending'
					if output_dir:
						_save_plan_to_md(plan, output_dir)
					return {"stop_requested": True, "research_plan": plan}

				logger.info(f"Executing tool: {tool_name}")
				tool_output = await selected_tool.ainvoke(tool_args)
				logger.info(f"Tool '{tool_name}' executed successfully.")
				browser_tool_called = "parallel_browser_search" in executed_tool_names
				if browser_tool_called:
					current_search_results.extend(tool_output)
				else:
					logger.info(f"Result from tool '{tool_name}': {str(tool_output)[:200]}...")

				tool_results.append(ToolMessage(
					content=json.dumps(tool_output),
					tool_call_id=tool_call_id
				))

			except Exception as e:
				logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
				tool_results.append(ToolMessage(
					content=f"Error executing tool {tool_name}: {e}",
					tool_call_id=tool_call_id
				))
				current_search_results.append({"tool_name": tool_name, "args": tool_args, "status": "failed", "error": str(e)})

		browser_tool_called = "parallel_browser_search" in executed_tool_names
		step_failed = any("Error:" in str(tr.content) for tr in tool_results) or not browser_tool_called

		if step_failed:
			logger.warning(f"Step {current_step['step']} failed or did not yield results via browser search.")
			current_step['status'] = 'failed'
			current_step['result_summary'] = (
				f"Tool execution failed or browser tool not used. Errors: "
				f"{[tr.content for tr in tool_results if 'Error' in str(tr.content)]}"
			)
		else:
			logger.info(f"Step {current_step['step']} completed using tool(s): {executed_tool_names}.")
			current_step['status'] = 'completed'
			current_step['result_summary'] = f"Executed tool(s): {', '.join(executed_tool_names)}."

		if output_dir:
			_save_plan_to_md(plan, output_dir)
			_save_search_results_to_json(current_search_results, output_dir)

		return {
			"research_plan": plan,
			"search_results": current_search_results,
			"current_step_index": current_index + 1,
			"messages": state["messages"] + current_task_message + [ai_response] + tool_results,
		}

	except Exception as e:
		logger.error(f"Unhandled error during research execution node for step {current_step['step']}: {e}", exc_info=True)
		current_step['status'] = 'failed'
		if output_dir:
			_save_plan_to_md(plan, output_dir)
		return {
			"research_plan": plan,
			"current_step_index": current_index + 1,
			"error_message": f"Core Execution Error on step {current_step['step']}: {e}"
		}


async def synthesis_node(state: DeepResearchState) -> Dict[str, Any]:
	"""Synthesizes the final report from the collected search results."""
	logger.info("--- Entering Synthesis Node ---")
	if state.get('stop_requested'):
		logger.info("Stop requested, skipping synthesis.")
		return {"stop_requested": True}

	llm = state['llm']
	topic = state['topic']
	search_results = state.get('search_results', [])
	output_dir = state['output_dir']
	plan = state['research_plan']

	if not search_results:
		logger.warning("No search results found to synthesize report.")
		report = f"# Research Report: {topic}\n\nNo information was gathered during the research process."
		if output_dir:
			_save_report_to_md(report, output_dir)
		return {"final_report": report}

	logger.info(f"Synthesizing report from {len(search_results)} collected search result entries.")

	formatted_results = ""
	for result_entry in search_results:
		query = result_entry.get('query', 'Unknown Query')
		status = result_entry.get('status', 'unknown')
		result_data = result_entry.get('result')
		error = result_entry.get('error')

		if status == 'completed' and result_data:
			summary = result_data
			formatted_results += f"### Finding from Query: \"{query}\"\n"
			formatted_results += f"- Summary:\n{summary}\n"
			formatted_results += "---\n"
		elif status == 'failed':
			formatted_results += f"### Failed Query: \"{query}\"\n"
			formatted_results += f"- Error: {error}\n"
			formatted_results += "---\n"

	plan_summary = "\nResearch Plan Followed:\n"
	for item in plan:
		marker = "- [x]" if item['status'] == 'completed' else "- [ ] (Failed)" if item['status'] == 'failed' else "- [ ]"
		plan_summary += f"{marker} {item['task']}\n"

	synthesis_prompt = ChatPromptTemplate.from_messages([
		("system", """You are a professional researcher tasked with writing a comprehensive and well-structured report based on collected findings.
		The report should address the research topic thoroughly, synthesizing the information gathered from various sources.
		Structure the report logically:
		1.  Introduction: Briefly introduce the topic and the report's scope.
		2.  Main Body: Discuss the key findings, organizing them thematically or according to the research plan steps. Analyze, compare, and contrast information from different sources where applicable. Cite sources implicitly by referencing queries.
		3.  Conclusion: Summarize the main points and offer concluding thoughts or potential areas for further research.
		Ensure the tone is objective, professional, and analytical. Base the report strictly on the provided findings.
		"""),
		("human", f"""
		Research Topic: {topic}

		{plan_summary}

		Collected Findings:
		```
		{formatted_results}
		```

		Please generate the final research report in Markdown format based only on the information above.
		""")
	])

	try:
		response = await llm.ainvoke(synthesis_prompt.format_prompt(
			topic=topic,
			plan_summary=plan_summary,
			formatted_results=formatted_results,
		).to_messages())
		final_report_md = response.content
		logger.info("Successfully synthesized the final report.")
		if output_dir:
			_save_report_to_md(final_report_md, output_dir)
		return {"final_report": final_report_md}

	except Exception as e:
		logger.error(f"Error during report synthesis: {e}", exc_info=True)
		return {"error_message": f"LLM Error during synthesis: {e}"}


def should_continue(state: DeepResearchState) -> str:
	"""Determines the next step based on the current state."""
	logger.info("--- Evaluating Condition: Should Continue? ---")
	if state.get('stop_requested'):
		logger.info("Stop requested, routing to END.")
		return "end_run"
	if state.get('error_message'):
		logger.warning(f"Error detected: {state['error_message']}. Routing to END.")
		return "end_run"

	plan = state.get('research_plan')
	current_index = state.get('current_step_index', 0)

	if not plan:
		logger.warning("No research plan found, cannot continue execution. Routing to END.")
		return "end_run"

	if current_index < len(plan):
		logger.info(f"Plan has pending steps (current index {current_index}/{len(plan)}). Routing to Research Execution.")
		return "execute_research"
	else:
		logger.info("All plan steps processed. Routing to Synthesis.")
		return "synthesize_report"


class DeepResearchAgent:
	def __init__(self, llm: Any, browser_config: Dict[str, Any], mcp_server_config: Optional[Dict[str, Any]] = None):
		"""
		Initializes the DeepResearchAgent.

		Args:
			llm: The Langchain compatible language model instance.
			browser_config: Configuration dictionary for the BrowserUseAgent tool.
			mcp_server_config: Optional configuration for the MCP client.
		"""
		self.llm = llm
		self.browser_config = browser_config
		self.mcp_server_config = mcp_server_config
		self.mcp_client = None
		self.stopped = False
		self.graph = self._compile_graph()
		self.current_task_id: Optional[str] = None
		self.stop_event: Optional[threading.Event] = None
		self.runner: Optional[asyncio.Task] = None

	async def _setup_tools(self, task_id: str, stop_event: threading.Event, max_tabs: int = 1) -> List[Tool]:
		"""Sets up the basic tools (File I/O) and optional MCP tools."""
		tools = [WriteFileTool(), ReadFileTool(), ListDirectoryTool()]
		browser_use_tool = create_browser_search_tool(
			llm=self.llm,
			browser_config=self.browser_config,
			task_id=task_id,
			stop_event=stop_event,
			max_tabs=max_tabs
		)
		tools += [browser_use_tool]
		if self.mcp_server_config:
			try:
				logger.info("Setting up MCP client and tools...")
				if not self.mcp_client:
					self.mcp_client = await setup_mcp_client_and_tools(self.mcp_server_config)
				client = self.mcp_client
				if client:
					mcp_tools = client.get_tools()
					logger.info(f"Loaded {len(mcp_tools)} MCP tools.")
					tools.extend(mcp_tools)
			except Exception as e:
				logger.error(f"Failed to set up MCP tools: {e}", exc_info=True)
		tools_map = {tool.name: tool for tool in tools}
		return list(tools_map.values())

	async def close_mcp_client(self):
		if self.mcp_client:
			await self.mcp_client.__aexit__(None, None, None)
			self.mcp_client = None

	def _compile_graph(self) -> StateGraph:
		"""Compiles the Langgraph state machine."""
		workflow = StateGraph(DeepResearchState)

		workflow.add_node("plan_research", planning_node)
		workflow.add_node("execute_research", research_execution_node)
		workflow.add_node("synthesize_report", synthesis_node)
		workflow.add_node("end_run", lambda state: logger.info("--- Reached End Run Node ---") or {})

		workflow.set_entry_point("plan_research")
		workflow.add_edge("plan_research", "execute_research")

		workflow.add_conditional_edges(
			"execute_research",
			should_continue,
			{
				"execute_research": "execute_research",
				"synthesize_report": "synthesize_report",
				"end_run": "end_run"
			}
		)

		workflow.add_edge("synthesize_report", "end_run")

		app = workflow.compile()
		return app

	async def run(self, topic: str, save_dir: Optional[str] = None, task_id: Optional[str] = None, max_parallel_browsers: int = 1, *, max_tabs: Optional[int] = None, max_windows: Optional[int] = None) -> Dict[str, Any]:
		"""
		Starts the deep research process.

		Args:
			topic: The research topic.
			save_dir: Optional directory to save outputs for this task. If None, operates in memory-only mode.
			task_id: Optional existing task ID to resume. If None, a new ID is generated.
			max_parallel_browsers: Max parallel workers (legacy). Deprecated: prefer max_tabs.
			max_tabs: Preferred naming for concurrency—max concurrent tabs within one window.
			max_windows: Deprecated alias retained for backward compatibility.

		Returns:
			 A dictionary containing the final status, message, task_id, and final_state.
		"""
		if self.runner and not self.runner.done():
			logger.warning("Agent is already running. Please stop the current task first.")
			return {"status": "error", "message": "Agent already running.", "task_id": self.current_task_id}

		self.current_task_id = task_id if task_id else str(uuid.uuid4())
		output_dir: Optional[Path] = None

		if save_dir:
			output_dir = Path(save_dir) / self.current_task_id
			output_dir.mkdir(parents=True, exist_ok=True)
			logger.info(f"[AsyncGen] Output directory: {str(output_dir)}")
		else:
			logger.info("[AsyncGen] Running in memory-only mode (no save_dir provided)")

		logger.info(f"[AsyncGen] Starting research task ID: {self.current_task_id} for topic: '{topic}'")

		self.stop_event = threading.Event()
		_AGENT_STOP_FLAGS[self.current_task_id] = self.stop_event
		if max_tabs is not None:
			effective_max = max_tabs
		elif max_windows is not None:
			effective_max = max_windows
		else:
			effective_max = max_parallel_browsers
		agent_tools = await self._setup_tools(self.current_task_id, self.stop_event, effective_max)
		initial_state: DeepResearchState = {
			"task_id": self.current_task_id,
			"topic": topic,
			"research_plan": [],
			"search_results": [],
			"messages": [],
			"llm": self.llm,
			"tools": agent_tools,
			"output_dir": output_dir,
			"browser_config": self.browser_config,
			"final_report": None,
			"current_step_index": 0,
			"stop_requested": False,
			"error_message": None,
		}

		loaded_state: Dict[str, Any] = {}
		if task_id and output_dir:
			logger.info(f"Attempting to resume task {task_id}...")
			loaded_state = _load_previous_state(task_id, str(output_dir))
			for key in [
				"research_plan",
				"current_step_index",
				"search_results",
				"error_message",
			]:
				if key in loaded_state:
					initial_state[key] = loaded_state[key]
			if loaded_state.get("research_plan"):
				logger.info(
					f"Resuming with {len(loaded_state['research_plan'])} plan steps and {len(loaded_state.get('search_results', []))} existing results.")
				initial_state["topic"] = topic
			else:
				logger.warning(f"Resume requested for {task_id}, but no previous plan found. Starting fresh.")
				initial_state["current_step_index"] = 0

		final_state = None
		status = "unknown"
		message = None
		try:
			logger.info(f"Invoking graph execution for task {self.current_task_id}...")
			self.runner = asyncio.create_task(self.graph.ainvoke(initial_state))
			final_state = await self.runner
			logger.info(f"Graph execution finished for task {self.current_task_id}.")

			if self.stop_event and self.stop_event.is_set():
				status = "stopped"
				message = "Research process was stopped by request."
				logger.info(message)
			elif final_state and final_state.get("error_message"):
				status = "error"
				message = final_state["error_message"]
				logger.error(f"Graph execution completed with error: {message}")
			elif final_state and final_state.get("final_report"):
				status = "completed"
				message = "Research process completed successfully."
				logger.info(message)
			else:
				status = "finished_incomplete"
				message = "Research process finished, but may be incomplete (no final report generated)."
				logger.warning(message)

		except asyncio.CancelledError:
			status = "cancelled"
			message = f"Agent run task cancelled for {self.current_task_id}."
			logger.info(message)
		except Exception as e:
			status = "error"
			message = f"Unhandled error during graph execution for {self.current_task_id}: {e}"
			logger.error(message, exc_info=True)
		finally:
			logger.info(f"Cleaning up resources for task {self.current_task_id}")
			task_id_to_clean = self.current_task_id

			self.stop_event = None
			self.current_task_id = None
			self.runner = None
			if self.mcp_client:
				await self.mcp_client.__aexit__(None, None, None)

			result = {
				"status": status,
				"message": message,
				"task_id": task_id_to_clean,
				"final_state": final_state if final_state else {}
			}

			if output_dir and final_state and final_state.get("final_report"):
				report_path = output_dir / REPORT_FILENAME
				result["report_file_path"] = str(report_path)

			return result

	async def _stop_lingering_browsers(self, task_id):
		"""Attempts to stop any BrowserUseAgent instances associated with the task_id."""
		keys_to_stop = [key for key in _BROWSER_AGENT_INSTANCES if key.startswith(f"{task_id}_")]
		if not keys_to_stop:
			return

		logger.warning(
			f"Found {len(keys_to_stop)} potentially lingering browser agents for task {task_id}. Attempting stop...")
		for key in keys_to_stop:
			agent_instance = _BROWSER_AGENT_INSTANCES.get(key)
			try:
				if agent_instance:
					await agent_instance.stop()
					logger.info(f"Called stop() on browser agent instance {key}")
			except Exception as e:
				logger.error(f"Error calling stop() on browser agent instance {key}: {e}")

	async def stop(self):
		"""Signals the currently running agent task to stop."""
		if not self.current_task_id or not self.stop_event:
			logger.info("No agent task is currently running.")
			return

		logger.info(f"Stop requested for task ID: {self.current_task_id}")
		self.stop_event.set()
		self.stopped = True
		await self._stop_lingering_browsers(self.current_task_id)

	def close(self):
		self.stopped = False
