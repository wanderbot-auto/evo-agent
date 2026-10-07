import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from openai import APITimeoutError
import httpx

import index
import runtime
from runtime import TraceModel, mcp_call, visible_text
from smolagents.models import ChatMessage


def test_protected_api_and_input_limits(monkeypatch):
    monkeypatch.setenv('APP_ACCESS_TOKEN','test-access')
    client=TestClient(index.app)
    assert client.get('/api/models').status_code==401
    headers={'Authorization':'Bearer test-access'}
    assert client.get('/api/models',headers=headers).json()['default_model']=='MiniMax-M3'
    assert client.post('/api/run',headers=headers,json={'task':'hi','max_steps':100}).status_code==422
    assert client.post('/api/run',headers=headers,json={'task':'hi','model':'unverified'}).status_code==400


def test_sse_failure_scrubs_secret(monkeypatch):
    monkeypatch.setenv('APP_ACCESS_TOKEN','test-access')
    monkeypatch.setenv('OPENAI_API_KEY','not-a-real-test-secret')
    def failing_run(payload,emit,cancelled):
        emit('run.start','开始',{})
        raise RuntimeError('not-a-real-test-secret')
    monkeypatch.setattr(index,'run_agent',failing_run)
    with TestClient(index.app) as client:
        response=client.post('/api/run',headers={'Authorization':'Bearer test-access'},json={'task':'hi'})
    events=[json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
    assert events[-1]['type']=='run.error'
    assert 'not-a-real-test-secret' not in response.text
    assert '[redacted]' in response.text


def test_compression_preserves_complete_latest_group():
    model=TraceModel.__new__(TraceModel)
    model.model_id='MiniMax-M3.1-Flash-Preview'
    model.cancelled=threading.Event()
    model.deadline=float('inf')
    model.input_tokens=model.output_tokens=0
    model.budget=200
    events=[]
    model.emit=lambda kind,title,data:events.append((kind,data))
    model.history=[{'role':'system','content':'rules'},{'role':'user','content':'task'},
                   {'role':'assistant','content':'x'*2000,'tool_calls':[{'id':'old'}]},
                   {'role':'tool','tool_call_id':'old','content':'fact'},
                   {'role':'assistant','content':'middle','tool_calls':[{'id':'middle'}]},
                   {'role':'tool','tool_call_id':'middle','content':'preserved'},
                   {'role':'assistant','content':'latest','reasoning_content':'private','tool_calls':[{'id':'new'}]},
                   {'role':'tool','tool_call_id':'new','content':'42'}]
    response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='摘要：fact'))],usage=SimpleNamespace(prompt_tokens=10,completion_tokens=5))
    model.request=lambda **kwargs:response
    model.compress()
    assert model.history[-2]['tool_calls'][0]['id']=='new'
    assert model.history[-1]['tool_call_id']=='new'
    assert model.history[-2]['reasoning_content']=='private'
    assert 'private' not in json.dumps(model.trace_history())
    assert events[-1][0]=='compression.end'
    assert events[-1][1]['estimated_tokens_after']<events[-1][1]['estimated_tokens_before']


def test_actual_mcp_stdio_handshake_and_tool():
    events=[]
    result=asyncio.run(mcp_call('memory',lambda kind,title,data:events.append(kind)))
    assert 'localStorage' in result
    assert events==['mcp.connect','mcp.discover','mcp.call','mcp.result']


def test_mcp_child_can_load_bundled_dependencies(monkeypatch):
    # A base interpreter does not automatically include the virtualenv site-packages.
    monkeypatch.setattr(runtime.sys,'executable',runtime.sys._base_executable)
    events=[]
    result=asyncio.run(mcp_call('mcp',lambda kind,title,data:events.append(kind)))
    assert 'initialize' in result
    assert 'mcp.result' in events


def test_thinking_not_exported():
    assert visible_text('<think>private reasoning</think>answer')=='answer'
    assert visible_text('<think>incomplete')==''


def test_step_limit_is_not_reported_as_success(monkeypatch):
    def repeat_multiply(self,messages,**kwargs):
        call={'id':'call_'+str(len(self.history)), 'type':'function','function':{'name':'multiply','arguments':'{"a":2,"b":3}'}}
        self.history.append({'role':'assistant','content':None,'tool_calls':[call]})
        return ChatMessage.from_dict({'role':'assistant','content':None,'tool_calls':[call]})
    monkeypatch.setattr(TraceModel,'generate',repeat_multiply)
    events=[]
    payload=SimpleNamespace(task='Repeated action',memory={},model='MiniMax-M3.1-Flash-Preview',context_budget=4000,max_steps=1)
    runtime.run_agent(payload,lambda kind,title,data:events.append((kind,data)),threading.Event())
    assert events[-1][0]=='run.incomplete'
    assert not any(kind=='run.end' for kind,data in events)


def test_repeated_tool_names_have_distinct_result_ids():
    model=TraceModel.__new__(TraceModel)
    model.lock=threading.Lock()
    model.ledger=[]
    model.history=[{'role':'assistant','tool_calls':[
        {'id':'a','function':{'name':'add','arguments':'{"a":1,"b":2}'}},
        {'id':'b','function':{'name':'add','arguments':'{"a":3,"b":4}'}},
    ]}]
    model.record_tool('add',3)
    model.record_tool('add',7)
    assert [message['tool_call_id'] for message in model.history[1:]]==['a','b']


def test_model_timeout_retries_once_then_reports_actionable_error():
    model=TraceModel.__new__(TraceModel)
    model.cancelled=threading.Event()
    model.deadline=float('inf')
    events=[]
    model.emit=lambda kind,title,data:events.append((kind,data))
    attempts=[]
    def timeout(**kwargs):
        attempts.append(kwargs)
        raise APITimeoutError(request=httpx.Request('POST','https://example.com'))
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=timeout)))
    model.client=SimpleNamespace(with_options=lambda **kwargs:client)
    with pytest.raises(APITimeoutError) as error:
        model.request(model='test',messages=[])
    assert len(attempts)==2
    assert events==[('model.retry',{'attempt':2,'code':'model_timeout','error':'MiniMax 请求超时，请稍后重试。'})]
    wrapper=RuntimeError('generation failed')
    wrapper.__cause__=error.value
    assert runtime.error_details(wrapper)['code']=='model_timeout'


def test_cancelled_run_does_not_retry_model():
    model=TraceModel.__new__(TraceModel)
    model.cancelled=threading.Event()
    model.cancelled.set()
    model.deadline=float('inf')
    with pytest.raises(RuntimeError,match='停止'):
        model.request()


def test_server_deadline_returns_terminal_sse_event(monkeypatch):
    monkeypatch.setenv('APP_ACCESS_TOKEN','test-access')
    monkeypatch.setattr(index,'RUN_TIMEOUT',0)
    def wait_for_cancellation(payload,emit,cancelled):
        cancelled.wait(2)
    monkeypatch.setattr(index,'run_agent',wait_for_cancellation)
    with TestClient(index.app) as client:
        response=client.post('/api/run',headers={'Authorization':'Bearer test-access'},json={'task':'hi'})
    events=[json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
    assert events[-1]['type']=='run.error'
    assert events[-1]['data']['code']=='run_timeout'
