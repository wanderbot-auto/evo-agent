import asyncio
import hmac
import json
import os
import threading
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

ROOT=Path(__file__).parent
load_dotenv(ROOT/'.env',override=False)

from runtime import RUN_TIMEOUT, error_details, run_agent

CATALOG=json.loads((ROOT/'models.json').read_text())
AVAILABLE={m['id'] for m in CATALOG['models'] if m['available'] and m['tool_call']}
app=FastAPI(title='Evo Agent',docs_url=None,redoc_url=None,openapi_url=None)

class RunRequest(BaseModel):
    task:str=Field(min_length=1,max_length=6000)
    model:str=CATALOG['default_model']
    max_steps:int=Field(default=10,ge=1,le=16)
    context_budget:int=Field(default=4000,ge=200,le=20000)
    memory:dict[str,str]=Field(default_factory=dict)

    @field_validator('memory')
    @classmethod
    def bound_memory(cls,value):
        if len(value)>30 or any(len(k)>100 or len(v)>2000 for k,v in value.items()):
            raise ValueError('记忆大小超出限制')
        return value

def authorize(authorization):
    token=os.getenv('APP_ACCESS_TOKEN','')
    if not token:
        raise HTTPException(503,'未配置应用访问口令')
    if not hmac.compare_digest(authorization or '', 'Bearer '+token):
        raise HTTPException(401,'访问口令不正确')

@app.get('/')
def home():
    return HTMLResponse((ROOT/'ui.html').read_text(),headers={'Cache-Control':'no-store'})

@app.get('/api/health')
def health():
    return {'status':'ok','model':CATALOG['default_model'],'configured':bool(os.getenv('OPENAI_API_KEY') and os.getenv('APP_ACCESS_TOKEN'))}

@app.get('/api/models')
def models(authorization:str|None=Header(default=None)):
    authorize(authorization)
    return CATALOG

@app.post('/api/run')
async def run(payload:RunRequest,request:Request,authorization:str|None=Header(default=None)):
    authorize(authorization)
    if payload.model not in AVAILABLE:
        raise HTTPException(400,'模型未通过可用性与工具调用验证')
    queue=asyncio.Queue()
    loop=asyncio.get_running_loop()
    cancelled=threading.Event()
    run_id=uuid.uuid4().hex[:12]
    start=time.monotonic()
    counter=0
    lock=threading.Lock()
    def make_event(kind,title,data):
        nonlocal counter
        with lock:
            counter+=1
            return {'id':counter,'run_id':run_id,'type':kind,'title':title,'elapsed_ms':round((time.monotonic()-start)*1000),'data':data}
    def emit(kind,title,data):
        if not cancelled.is_set():
            loop.call_soon_threadsafe(queue.put_nowait,make_event(kind,title,data))
    def worker():
        try:
            run_agent(payload,emit,cancelled)
        except Exception as exc:
            emit('run.error','执行失败',error_details(exc))
        finally:
            if not cancelled.is_set(): loop.call_soon_threadsafe(queue.put_nowait,None)
    async def stream():
        worker_task=asyncio.create_task(asyncio.to_thread(worker))
        try:
            yield ': connected\n\n'
            while True:
                if await request.is_disconnected():
                    cancelled.set();break
                if time.monotonic()-start>=RUN_TIMEOUT:
                    event=make_event('run.error','运行超时',{'code':'run_timeout','error':'运行超过 220 秒，请缩短任务后重试。'})
                    yield 'data: '+json.dumps(event,ensure_ascii=False)+'\n\n'
                    break
                try:
                    event=await asyncio.wait_for(queue.get(),timeout=3)
                except asyncio.TimeoutError:
                    yield ': heartbeat\n\n';continue
                if event is None: break
                yield 'data: '+json.dumps(event,ensure_ascii=False,default=str)+'\n\n'
        finally:
            cancelled.set()
            # Keep a reference until the worker exits; cancellation checked each chunk/step.
            worker_task.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
    return StreamingResponse(stream(),media_type='text/event-stream',headers={'Cache-Control':'no-cache, no-transform','X-Accel-Buffering':'no'})
