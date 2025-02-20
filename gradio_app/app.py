import gradio as gr
import os
import sys
from pathlib import Path
import traceback
from typing import Optional, Generator, AsyncGenerator
import asyncio
import queue
import logging
import io
from datetime import datetime
import requests
from dotenv import load_dotenv, set_key

# Add the parent directory to sys.path to import mlx_use
sys.path.append(str(Path(__file__).parent.parent))

from mlx_use import Agent
from mlx_use.controller.service import Controller
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

# Set up logging to capture terminal output
class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_queue.put(f"{msg}\n")
        except Exception:
            self.handleError(record)

# LLM model mappings
LLM_MODELS = {
    "OpenAI": ["gpt-4o", "o3-mini"],
    "Anthropic": ["claude-3-5-sonnet-20240620"],
    "Google": ["gemini-1.5-flash-002"],
    "alibaba": ["qwen-2.5-72b-instruct"]
}

def get_llm(provider: str, model: str, api_key: str) -> Optional[object]:
    """Initialize LLM based on provider"""
    try:
        if provider == "OpenAI":
            return ChatOpenAI(model=model, api_key=SecretStr(api_key))
        elif provider == "Anthropic":
            return ChatAnthropic(model=model, api_key=SecretStr(api_key))
        elif provider == "Google":
            return ChatGoogleGenerativeAI(model=model, api_key=SecretStr(api_key))
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    except Exception as e:
        raise ValueError(f"Failed to initialize {provider} LLM: {str(e)}")

class MacOSUseGradioApp:
    def __init__(self):
        self.agent = None
        self.controller = Controller()
        self.is_running = False
        self.log_queue = queue.Queue()
        self.setup_logging()
        self.terminal_buffer = []
        self.automations = {}  # Dictionary to store automation flows
        self._cleanup_state()
        
        # Load environment variables
        load_dotenv()

    def _cleanup_state(self):
        """Reset all state variables"""
        self.is_running = False
        self.agent = None
        self.terminal_buffer = []
        while not self.log_queue.empty():
            try:
                self.log_queue.get_nowait()
            except queue.Empty:
                break

    def setup_logging(self):
        """Set up logging to capture terminal output"""
        # Remove existing handlers to prevent duplicates
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create queue handler
        queue_handler = QueueHandler(self.log_queue)
        queue_handler.setFormatter(logging.Formatter('%(message)s'))
        root_logger.addHandler(queue_handler)

    def get_terminal_output(self) -> str:
        """Get accumulated terminal output"""
        while True:
            try:
                log = self.log_queue.get_nowait()
                self.terminal_buffer.append(log)
            except queue.Empty:
                break
        return "".join(self.terminal_buffer)

    def stream_terminal_output(self) -> Generator[str, None, None]:
        """Stream terminal output in real-time"""
        while self.is_running:
            output = self.get_terminal_output()
            if output:
                yield output
            yield

    def stop_agent(self) -> tuple:
        """Stop the running agent"""
        if self.agent and self.is_running:
            self.is_running = False
            return (
                self.get_terminal_output() + "\nAgent stopped by user",
                gr.update(interactive=True),
                gr.update(interactive=False)
            )
        return (
            "No agent running",
            gr.update(interactive=True),
            gr.update(interactive=False)
        )

    def update_model_choices(self, provider):
        """Update model choices based on provider selection"""
        return gr.Dropdown.update(choices=LLM_MODELS.get(provider, []))

    def add_automation(self, name: str, description: str) -> dict:
        """Add a new automation flow"""
        if name in self.automations:
            raise ValueError(f"Automation '{name}' already exists")
        
        self.automations[name] = {
            "description": description,
            "agents": []
        }
        return gr.update(choices=list(self.automations.keys()))

    def add_agent_to_automation(self, automation_name: str, agent_prompt: str, position: int = -1) -> list:
        """Add a new agent to an automation flow"""
        if automation_name not in self.automations:
            raise ValueError(f"Automation '{automation_name}' does not exist")
        
        new_agent = {
            "prompt": agent_prompt,
            "max_steps": 25,  # Default values
            "max_actions": 1
        }
        
        if position == -1 or position >= len(self.automations[automation_name]["agents"]):
            self.automations[automation_name]["agents"].append(new_agent)
        else:
            self.automations[automation_name]["agents"].insert(position, new_agent)
            
        return self.automations[automation_name]["agents"]

    def remove_agent_from_automation(self, automation_name: str, agent_index: int) -> list:
        """Remove an agent from an automation flow"""
        if automation_name not in self.automations:
            raise ValueError(f"Automation '{automation_name}' does not exist")
        
        if not isinstance(agent_index, int):
            raise ValueError("Agent index must be an integer")
            
        if agent_index < 0 or agent_index >= len(self.automations[automation_name]["agents"]):
            raise ValueError(f"Invalid agent index {agent_index}")
        
        self.automations[automation_name]["agents"].pop(agent_index)
        return self.automations[automation_name]["agents"]

    def update_agent_prompt(self, automation_name: str, agent_index: int, new_prompt: str) -> list:
        """Update an agent's prompt in an automation flow"""
        if automation_name not in self.automations:
            raise ValueError(f"Automation '{automation_name}' does not exist")
        
        if agent_index < 0 or agent_index >= len(self.automations[automation_name]["agents"]):
            raise ValueError(f"Invalid agent index {agent_index}")
        
        self.automations[automation_name]["agents"][agent_index]["prompt"] = new_prompt
        return self.automations[automation_name]["agents"]

    def get_automation_agents(self, automation_name: str) -> list:
        """Get the list of agents for an automation flow"""
        if automation_name not in self.automations:
            raise ValueError(f"Automation '{automation_name}' does not exist")
        
        return self.automations[automation_name]["agents"]

    def save_api_key_to_env(self, provider: str, api_key: str) -> None:
        """Save API key to .env file based on provider"""
        env_path = Path(__file__).parent.parent / '.env'
        
        # Create .env file if it doesn't exist
        if not env_path.exists():
            env_path.touch()
        
        # Map provider to environment variable name
        provider_to_env = {
            "OpenAI": "OPENAI_API_KEY",
            "Anthropic": "ANTHROPIC_API_KEY",
            "Google": "GEMINI_API_KEY",
            "alibaba": "DEEPSEEK_API_KEY"
        }
        
        env_var = provider_to_env.get(provider)
        if env_var and api_key:
            set_key(str(env_path), env_var, api_key)

    async def run_automation(
        self,
        automation_name: str,
        llm_provider: str,
        llm_model: str,
        api_key: str,
    ) -> AsyncGenerator[tuple[str, dict, dict], None]:
        """Run an automation flow by executing its agents in sequence"""
        if automation_name not in self.automations:
            raise ValueError(f"Automation '{automation_name}' does not exist")

        # Save API key to .env file
        self.save_api_key_to_env(llm_provider, api_key)

        automation = self.automations[automation_name]
        self._cleanup_state()
        
        try:
            for i, agent_config in enumerate(automation["agents"]):
                # Initialize LLM
                llm = get_llm(llm_provider, llm_model, api_key)
                if not llm:
                    raise ValueError(f"Failed to initialize {llm_provider} LLM")
                
                # Initialize agent
                self.agent = Agent(
                    task=agent_config["prompt"],
                    llm=llm,
                    controller=self.controller,
                    use_vision=False,
                    max_actions_per_step=agent_config["max_actions"]
                )
                
                self.is_running = True
                last_update = ""
                
                try:
                    # Start the agent run
                    agent_task = asyncio.create_task(self.agent.run(max_steps=agent_config["max_steps"]))
                    
                    # While the agent is running, yield updates periodically
                    while not agent_task.done() and self.is_running:
                        current_output = self.get_terminal_output()
                        if current_output != last_update:
                            yield (
                                f"Running agent {i+1}/{len(automation['agents'])}\n{current_output}",
                                gr.update(interactive=False),
                                gr.update(interactive=True)
                            )
                            last_update = current_output
                        await asyncio.sleep(0.1)
                    
                    if not agent_task.done():
                        agent_task.cancel()
                        await asyncio.sleep(0.1)  # Allow time for cancellation
                    else:
                        result = await agent_task
                    
                    # Final update for this agent
                    final_output = self.get_terminal_output()
                    if final_output != last_update:
                        yield (
                            f"Completed agent {i+1}/{len(automation['agents'])}\n{final_output}",
                            gr.update(interactive=True) if i == len(automation["agents"]) - 1 else gr.update(interactive=False),
                            gr.update(interactive=False) if i == len(automation["agents"]) - 1 else gr.update(interactive=True)
                        )
                    
                except Exception as e:
                    error_details = f"Error Details:\n{traceback.format_exc()}"
                    self.terminal_buffer.append(f"\nError occurred in agent {i+1}:\n{str(e)}\n\n{error_details}")
                    yield (
                        "".join(self.terminal_buffer),
                        gr.update(interactive=True),
                        gr.update(interactive=False)
                    )
                    break
                
                self._cleanup_state()
                
        except Exception as e:
            error_details = f"Error Details:\n{traceback.format_exc()}"
            error_msg = f"Error occurred:\n{str(e)}\n\n{error_details}"
            yield (
                error_msg,
                gr.update(interactive=True),
                gr.update(interactive=False)
            )
            self._cleanup_state()

    async def run_agent(
        self,
        task: str,
        max_steps: int,
        max_actions: int,
        llm_provider: str,
        llm_model: str,
        api_key: str,
        share_prompt: bool
    ) -> AsyncGenerator[tuple[str, dict, dict], None]:
        """Run the agent with the specified configuration and submit the prompt to the Google Sheet if requested."""
        # Clean up any previous state
        self._cleanup_state()
        
        try:
            # Validate inputs
            if not task:
                raise ValueError("Task description is required")
            if not api_key:
                raise ValueError("API key is required")
            
            # Save API key to .env file
            self.save_api_key_to_env(llm_provider, api_key)
            
            # Send the prompt to the Google Form/Sheet if requested
            if share_prompt:
                try:
                    success = await asyncio.to_thread(send_prompt_to_google_sheet, task)
                    if not success:
                        logging.warning("Failed to send prompt to Google Form")
                except Exception as e:
                    logging.error(f"Error sending prompt to Google Form: {e}")
            
            # Initialize LLM
            llm = get_llm(llm_provider, llm_model, api_key)
            if not llm:
                raise ValueError(f"Failed to initialize {llm_provider} LLM")
            
            # Initialize agent
            self.agent = Agent(
                task=task,
                llm=llm,
                controller=self.controller,
                use_vision=False,
                max_actions_per_step=max_actions
            )
            
            self.is_running = True
            last_update = ""
            
            try:
                # Start the agent run
                agent_task = asyncio.create_task(self.agent.run(max_steps=max_steps))
                
                # While the agent is running, yield updates periodically.
                while not agent_task.done() and self.is_running:
                    current_output = self.get_terminal_output()
                    if current_output != last_update:
                        yield (
                            current_output,
                            gr.update(interactive=False),
                            gr.update(interactive=True)
                        )
                        last_update = current_output
                    await asyncio.sleep(0.1)
                
                if not agent_task.done():
                    agent_task.cancel()
                    await asyncio.sleep(0.1)  # Allow time for cancellation
                else:
                    result = await agent_task
                
                # Final update
                final_output = self.get_terminal_output()
                if final_output != last_update:
                    yield (
                        final_output,
                        gr.update(interactive=True),
                        gr.update(interactive=False)
                    )
                
            except Exception as e:
                error_details = f"Error Details:\n{traceback.format_exc()}"
                self.terminal_buffer.append(f"\nError occurred:\n{str(e)}\n\n{error_details}")
                yield (
                    "".join(self.terminal_buffer),
                    gr.update(interactive=True),
                    gr.update(interactive=False)
                )
            finally:
                self._cleanup_state()
            
        except Exception as e:
            error_details = f"Error Details:\n{traceback.format_exc()}"
            error_msg = f"Error occurred:\n{str(e)}\n\n{error_details}"
            yield (
                error_msg,
                gr.update(interactive=True),
                gr.update(interactive=False)
            )
            self._cleanup_state()

    def get_saved_api_key(self, provider: str) -> str:
        """Get saved API key from .env file based on provider"""
        provider_to_env = {
            "OpenAI": "OPENAI_API_KEY",
            "Anthropic": "ANTHROPIC_API_KEY",
            "Google": "GEMINI_API_KEY",
            "alibaba": "DEEPSEEK_API_KEY"
        }
        env_var = provider_to_env.get(provider)
        return os.getenv(env_var, "") if env_var else ""

    def create_interface(self):
        """Create the Gradio interface with the share prompt checkbox."""
        with gr.Blocks(title="macOS-use Interface") as demo:
            gr.Markdown("# Make Mac apps accessible for AI agents(Beta)")
            
            with gr.Tab("Agent"):
                with gr.Row():
                    with gr.Column(scale=2):
                        task_input = gr.Textbox(
                            label="Task Description",
                            placeholder="Enter task (e.g., 'open calculator')",
                            lines=3
                        )
                        share_prompt = gr.Checkbox(
                            label="Share prompt (only!) anonymously",
                            value=False,
                            info="Sharing your prompt (and prompt only) ANONYMOUSLY will help us improve our agent."
                        )
                        with gr.Row():
                            max_steps = gr.Slider(
                                minimum=1,
                                maximum=100,
                                value=25,
                                step=1,
                                label="Max Run Steps"
                            )
                            max_actions = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=1,
                                step=1,
                                label="Max Actions per Step"
                            )
                        with gr.Row():
                            run_button = gr.Button("Run", variant="primary")
                            stop_button = gr.Button("Stop", interactive=False)
                    
                    with gr.Column(scale=3):
                        terminal_output = gr.Textbox(
                            label="Terminal Output",
                            lines=25,
                            interactive=False,
                            autoscroll=True
                        )

            with gr.Tab("Automations"):
                with gr.Row():
                    with gr.Column(scale=2):
                        automation_name = gr.Textbox(
                            label="Automation Name",
                            placeholder="Enter automation name"
                        )
                        automation_description = gr.Textbox(
                            label="Description",
                            placeholder="Enter automation description",
                            lines=2
                        )
                        add_automation_btn = gr.Button("Add Automation", variant="primary")
                        
                        automation_list = gr.Dropdown(
                            label="Select Automation",
                            choices=list(self.automations.keys()),
                            interactive=True
                        )
                        
                        agent_prompt = gr.Textbox(
                            label="Agent Prompt",
                            placeholder="Enter agent prompt",
                            lines=3,
                            interactive=True
                        )
                        
                        with gr.Row():
                            add_agent_btn = gr.Button("Add Agent", variant="primary")
                            remove_agent_btn = gr.Button("Remove Selected Agent", variant="stop")
                        
                        run_automation_btn = gr.Button("Run Automation", variant="primary")
                        
                    with gr.Column(scale=3):
                        agents_list = gr.List(
                            label="Agents in Flow",
                            headers=["#", "Prompt"],
                            type="array",
                            interactive=True,
                            col_count=2
                        )
                        automation_output = gr.Textbox(
                            label="Automation Output",
                            lines=25,
                            interactive=False,
                            autoscroll=True
                        )
            
            with gr.Tab("Configuration"):
                llm_provider = gr.Dropdown(
                    choices=list(LLM_MODELS.keys()),
                    label="LLM Provider",
                    value="OpenAI"
                )
                llm_model = gr.Dropdown(
                    choices=LLM_MODELS["OpenAI"],
                    label="Model"
                )
                api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    placeholder="Enter your API key",
                    value=self.get_saved_api_key("OpenAI")  # Load saved API key for default provider
                )
                
                # Update model choices and API key when provider changes
                def update_provider(provider):
                    return {
                        llm_model: gr.update(choices=LLM_MODELS.get(provider, [])),
                        api_key: gr.update(value=self.get_saved_api_key(provider))
                    }
                
                llm_provider.change(
                    fn=update_provider,
                    inputs=llm_provider,
                    outputs=[llm_model, api_key]
                )
            
            # Add automation event handlers
            def format_agents_list(agents):
                return [[i, agent["prompt"]] for i, agent in enumerate(agents)]

            def add_automation_and_clear(name: str, description: str):
                result = self.add_automation(name, description)
                return {
                    automation_list: result,
                    automation_name: "",
                    automation_description: ""
                }

            add_automation_btn.click(
                fn=add_automation_and_clear,
                inputs=[automation_name, automation_description],
                outputs=[automation_list, automation_name, automation_description]
            )

            def add_agent_and_update_list(automation_name, prompt):
                agents = self.add_agent_to_automation(automation_name, prompt)
                return {
                    agents_list: format_agents_list(agents),
                    agent_prompt: ""
                }

            add_agent_btn.click(
                fn=add_agent_and_update_list,
                inputs=[automation_list, agent_prompt],
                outputs=[agents_list, agent_prompt]
            )

            def remove_agent_and_update_list(automation_name, agent_index):
                if not agent_index or not isinstance(agent_index, list):
                    return format_agents_list(self.get_automation_agents(automation_name))
                try:
                    # Extract the index from the selected row
                    index = int(agent_index[0][0])  # First row, first column (index)
                    agents = self.remove_agent_from_automation(automation_name, index)
                    return format_agents_list(agents)
                except (ValueError, IndexError, TypeError):
                    return format_agents_list(self.get_automation_agents(automation_name))

            remove_agent_btn.click(
                fn=remove_agent_and_update_list,
                inputs=[automation_list, agents_list],
                outputs=[agents_list]
            )

            def update_agents_list(automation_name):
                if not automation_name:
                    return []
                agents = self.get_automation_agents(automation_name)
                return format_agents_list(agents)

            automation_list.change(
                fn=update_agents_list,
                inputs=[automation_list],
                outputs=[agents_list]
            )

            run_automation_btn.click(
                fn=self.run_automation,
                inputs=[
                    automation_list,
                    llm_provider,
                    llm_model,
                    api_key,
                ],
                outputs=[
                    automation_output,
                    run_automation_btn,
                    stop_button
                ],
                queue=True,
                api_name=False
            )

            # Existing event handlers
            run_button.click(
                fn=self.run_agent,
                inputs=[
                    task_input,
                    max_steps,
                    max_actions,
                    llm_provider,
                    llm_model,
                    api_key,
                    share_prompt
                ],
                outputs=[
                    terminal_output,
                    run_button,
                    stop_button
                ],
                queue=True,
                api_name=False
            )
            
            stop_button.click(
                fn=self.stop_agent,
                outputs=[
                    terminal_output,
                    run_button,
                    stop_button
                ]
            )

        return demo

def send_prompt_to_google_sheet(prompt: str) -> bool:
    """
    Sends the prompt text to a Google Form, which appends it to a linked Google Sheet.
    """
    form_url = "https://docs.google.com/forms/d/1kbAdjvIU3KCplgk5OhzyK9aW4WsQYp4NdqxelhMvkv4/formResponse"
    payload = {
        "entry.1235837381": prompt,
        "fvv": "1"
    }
    try:
        response = requests.post(form_url, data=payload)
        return response.status_code == 200
    except Exception as e:
        logging.error(f"Failed to send prompt to Google Form: {str(e)}")
        return False

def main():
    app = MacOSUseGradioApp()
    demo = app.create_interface()
    demo.queue(default_concurrency_limit=1)  # Limit to one concurrent task
    demo.launch(
        server_name="0.0.0.0",
        server_port=os.getenv("SERVER_PORT", 7860),
        inbrowser=os.getenv("LAUNCH_BROWSER", False),
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main() 
