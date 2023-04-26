import re
from typing import Optional, Tuple, Generator, List, Any
from queue import Queue
import threading

from langchain import OpenAI
from langchain.llms import GPT4All
from langchain.llms.base import LLM
import logging
from vocode import getenv

from vocode.streaming.agent.base_agent import BaseAgent
from vocode.streaming.agent.utils import stream_llm_response
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from vocode.streaming.models.agent import LLMAgentConfig

class FancyDict(dict):
    def to_dict(self):
        return self
def stream(model, *args, **kwargs):
    # Create a queue to store the tokens
    q = Queue()
    
    # Define a custom callback handler that puts tokens into the queue
    class QueueCallbackHandler(StreamingStdOutCallbackHandler):
        def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
            print("<START>" + token + "<END>")
            q.put(token)
    
    # Create a callback manager with the custom handler
    callback_manager = QueueCallbackHandler()
    model.callback_manager = callback_manager
    
    # Run the model in a separate thread with the callback manager
    t = threading.Thread(target=model, args=args, kwargs=kwargs)
    t.start()
    
    # Yield tokens from the queue until the thread is done
    while t.is_alive() or not q.empty():
        yield FancyDict({"choices": [{"text": q.get(), "finish_reason": None}]})

class LLMAgent(BaseAgent):
    SENTENCE_ENDINGS = [".", "!", "?"]

    DEFAULT_PROMPT_TEMPLATE = "{history}\nHuman: {human_input}\nAI:"

    def __init__(
        self,
        llm: LLM,
        agent_config: LLMAgentConfig,
        logger: logging.Logger = None,
        sender="AI",
        recipient="Human",
    ):
        super().__init__(agent_config)
        self.agent_config = agent_config
        self.prompt_template = (
            f"{agent_config.prompt_preamble}\n\n{self.DEFAULT_PROMPT_TEMPLATE}"
        )
        self.initial_bot_message = (
            agent_config.initial_message.text if agent_config.initial_message else None
        )
        self.logger = logger or logging.getLogger(__name__)
        self.sender = sender
        self.recipient = recipient
        self.memory = (
            [f"AI: {agent_config.initial_message.text}"]
            if agent_config.initial_message
            else []
        )
        self.llm = llm
        self.stop_tokens = [f"{recipient}:"]
        self.first_response = (
            self.llm(
                self.prompt_template.format(
                    history="", human_input=agent_config.expected_first_prompt
                ),
                stop=self.stop_tokens,
            ).strip()
            if agent_config.expected_first_prompt
            else None
        )
        self.is_first_response = True

    def create_prompt(self, human_input):
        history = "\n".join(self.memory[-5:])
        return self.prompt_template.format(history=history, human_input=human_input)

    def get_memory_entry(self, human_input, response):
        return f"{self.recipient}: {human_input}\n{self.sender}: {response}"

    def respond(
        self,
        human_input,
        is_interrupt: bool = False,
        conversation_id: Optional[str] = None,
    ) -> Tuple[str, bool]:
        if is_interrupt and self.agent_config.cut_off_response:
            cut_off_response = self.get_cut_off_response()
            self.memory.append(self.get_memory_entry(human_input, cut_off_response))
            return cut_off_response, False
        self.logger.debug("LLM responding to human input")
        if self.is_first_response and self.first_response:
            self.logger.debug("First response is cached")
            self.is_first_response = False
            response = self.first_response
        else:
            response = self.llm(self.create_prompt(human_input), stop=self.stop_tokens)
            response = response.replace(f"{self.sender}:", "")
        self.memory.append(self.get_memory_entry(human_input, response))
        self.logger.debug(f"LLM response: {response}")
        return response, False

    def generate_response(
        self,
        human_input,
        is_interrupt: bool = False,
        conversation_id: Optional[str] = None,
    ) -> Generator:
        self.logger.debug("LLM generating response to human input")
        if is_interrupt and self.agent_config.cut_off_response:
            cut_off_response = self.get_cut_off_response()
            self.memory.append(self.get_memory_entry(human_input, cut_off_response))
            yield cut_off_response
            return
        self.memory.append(self.get_memory_entry(human_input, ""))
        if self.is_first_response and self.first_response:
            self.logger.debug("First response is cached")
            self.is_first_response = False
            sentences = [self.first_response]
        else:
            self.logger.debug("Creating LLM prompt")
            prompt = self.create_prompt(human_input)
            self.logger.debug("Streaming LLM response")
            sentences = stream_llm_response(
                map(
                    lambda resp: resp.to_dict(),
                    stream(self.llm, prompt, stop=self.stop_tokens),
                )
            )
        response_buffer = ""
        for sentence in sentences:
            sentence = sentence.replace(f"{self.sender}:", "")
            sentence = re.sub(r"^\s+(.*)", r" \1", sentence)
            response_buffer += sentence
            self.memory[-1] = self.get_memory_entry(human_input, response_buffer)
            yield sentence

    def update_last_bot_message_on_cut_off(self, message: str):
        last_message = self.memory[-1]
        new_last_message = (
            last_message.split("\n", 1)[0] + f"\n{self.sender}: {message}"
        )
        self.memory[-1] = new_last_message


if __name__ == "__main__":
    chat_responder = LLMAgent(GPT4All(
            #temperature=self.agent_config.temperature,
            #max_tokens=self.agent_config.max_tokens,
            n_ctx=512, 
            n_threads=8,
            model="/Users/zaptrem/dalai/llama/models/7B/gpt4all-lora-q-converted.bin",
        ),
        LLMAgentConfig(
            prompt_preamble="""
The AI is having a pleasant conversation about life.

{history}
Human: {human_input}
AI:""",
        )
    )
    while True:
        # response = chat_responder.respond(input("Human: "))[0]
        for response in chat_responder.generate_response(input("Human: ")):
            print(f"AI: {response}")
