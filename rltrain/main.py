from transformers.utils import get_json_schema
from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema, ToolResponse

class WeatherTool(BaseTool):
    def get_current_temperature(self, location: str, unit: str = "celsius"):
        """Return fake temperature for a location."""
        return {
            "temperature": 26.1,
            "location": location,
            "unit": unit,
        }

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        schema = get_json_schema(self.get_current_temperature)
        return OpenAIFunctionToolSchema(**schema)

    async def execute(self, instance_id: str, parameters: dict, **kwargs) -> tuple[ToolResponse, float, dict]:
        try:
            result = self.get_current_temperature(**parameters)
            return ToolResponse(text=json.dumps(result)), 0.0, {}
        except Exception as e:
            return ToolResponse(text=str(e)), 0.0, {}

weather_tool = WeatherTool(config={}, tool_schema=None)









from hydra import compose, initialize_config_dir
from verl.workers.rollout.replica import get_rollout_replica_class

with initialize_config_dir(config_dir=verl_config_dir):
    config = compose(
        config_name="ppo_trainer",
        overrides=[
            # Inference parameters
            f"actor_rollout_ref.rollout.name={rollout_name}",
            "actor_rollout_ref.rollout.mode=async",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            f"actor_rollout_ref.model.path={model_path}",
            "actor_rollout_ref.rollout.response_length=4096",
            # Enable automatic tool choice and parser
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=True",
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=hermes",
            "+actor_rollout_ref.rollout.engine_kwargs.sglang.tool_call_parser=qwen25",
        ],
    )

rollout_server_class = get_rollout_replica_class(config.actor_rollout_ref.rollout.name)
rollout_server = rollout_server_class(
    replica_rank=0,
    config=config.actor_rollout_ref.rollout,
    model_config=config.actor_rollout_ref.model,
)
await rollout_server.init_standalone()


from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="dummy", base_url=f"http://{rollout_server._server_address}/v1")
messages = [{"role": "user", "content": "Hey, what's the temperature in Paris right now?"}]

completion = await client.chat.completions.create(
    model=config.actor_rollout_ref.model.path,
    messages=messages,
    tools=[weather_tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)],
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
    },
)

message = completion.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
messages.append(message)
