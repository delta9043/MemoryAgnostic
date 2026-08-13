import uvloop
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.utils import cli_env_setup
from vllm.utils import FlexibleArgumentParser

"""
vLLM OpenAI 서버 런처.
"""

if __name__ == "__main__":
    cli_env_setup()
    parser = make_arg_parser(FlexibleArgumentParser(description="vLLM OpenAI server"))
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    # vllm 0.8.5 : JSON 사이 공백 금지 설정
    args.guided_decoding_backend = "xgrammar:disable-any-whitespace"

    uvloop.run(run_server(args))