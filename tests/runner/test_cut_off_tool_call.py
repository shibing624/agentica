from agentica.model.message import Message
from agentica.runner.loop import LoopMixin
from agentica.tools.base import get_function_call, Function


def _refused(call_id, content="Error while decoding function arguments"):
    return Message(role="tool", tool_call_id=call_id, tool_name="write_file", content=content, tool_call_error=True)


def test_a_call_cut_off_at_the_output_cap_is_told_to_split():
    call = {"id": "c1", "type": "function", "function": {"name": "write_file", "arguments": '{"content": "'}}
    assistant = Message(role="assistant", tool_calls=[call], finish_reason="length")
    messages = [Message(role="user", content="q"), assistant, _refused("c1")]
    LoopMixin._note_cut_off_tool_call(assistant, messages)
    assert "cut off" in messages[-1].content and "Do not resend it whole" in messages[-1].content


def test_a_malformed_call_that_was_not_cut_off_is_left_alone():
    call = {"id": "c1", "type": "function", "function": {"name": "write_file", "arguments": "{bad"}}
    assistant = Message(role="assistant", tool_calls=[call], finish_reason="tool_calls")
    refused = _refused("c1")
    LoopMixin._note_cut_off_tool_call(assistant, [assistant, refused])
    assert refused.content == "Error while decoding function arguments"


def test_an_undecodable_argument_string_is_echoed_only_in_part():
    def write_file(content: str) -> str:
        return content

    fn = Function.from_callable(write_file)
    huge = '{"content": "' + "}" * 50_000
    fc = get_function_call("write_file", huge, "c1", {"write_file": fn})
    assert fc.error.startswith("Error while decoding function arguments")
    assert len(fc.error) < 3_000
    assert f"({len(huge)} characters in total)" in fc.error
