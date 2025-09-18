#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Quickstart Example.

The example runs a simple voice AI bot that you can connect to using your
browser and speak with it. You can also deploy this bot to Pipecat Cloud.

Required AI services:
- Deepgram (Speech-to-Text)
- OpenAI (LLM)
- Cartesia (Text-to-Speech)

Run the bot using::

    uv run bot.py
"""

import os
import json
import time
import os
import uuid

from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import LLMRunFrame

from pipecat.audio.vad.silero import SileroVADAnalyzer

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

from pipecat.services.riva.stt import RivaSTTService
from pipecat.transcriptions.language import Language

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from pipecat.services.llm_service import FunctionCallParams

from openai.types.chat import ChatCompletionToolParam

from prompts import agent_prompts


load_dotenv(override=True)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    
    # stt = RivaSTTService(
    #     api_key=os.getenv("RIVA_API_KEY"),
    #     server=os.getenv("RIVA_SERVER"),
    #     params=RivaSTTService.InputParams(
    #         language=Language.EN_US
    #     )
    # )

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    # confirm_identity_fn = FunctionSchema(
    #     name="confirm_identity",
    #     description="Verify patient by ID and name; persist a short-lived dev session record.",
    #     properties={
    #         "patient_id": {"type": "string"},
    #         "last_name": {"type": "string"}
    #         # "dob": {"type": "string", "description": "YYYY-MM-DD"}
    #     },
    #     required=["patient_id", "last_name"],
    # )

    # list_active_meds_fn = FunctionSchema(
    #     name="list_active_medications",
    #     description="Return the active medications for a verified patient.",
    #     properties={"patient_id": {"type": "string"}},
    #     required=["patient_id"],
    # )

    # create_refill_invoice_fn = FunctionSchema(
    #     name="create_refill_invoice",
    #     description="Create a refill invoice for a given medication.",
    #     properties={
    #         "patient_id": {"type": "string"},
    #         "medication_id": {"type": "string", "description": "ID of the med to refill"},
    #         "quantity": {"type": "integer", "minimum": 1},
    #         "pharmacy": {"type": "string", "nullable": True}
    #     },
    #     required=["patient_id", "medication_id", "quantity"],
    # )
    # 
    # 
    # tools_schema_ = ToolsSchema(standard_tools=[confirm_identity_fn, list_active_meds_fn, create_refill_invoice_fn])

    tools_schema_ = [ 
        ChatCompletionToolParam( 
            type="function", 
            function={ 
                "name": "confirm_identity", 
                "description": "Ask for patient ID first. After they give their ID, then ask for their first name.",
                "parameters": { 
                    "type": "object", 
                    "properties": { 
                        "patient_id": {"type": "string"}, 
                        "first_name": {"type": "string"}
                          }, 
                        "required": ["patient_id", "first_name"], 
                        "additionalProperties": False 
                }, 
            }, 
        ), 
        ChatCompletionToolParam( 
            type="function", 
            function={ 
                "name": "list_active_medications", 
                "description": "Return the active medications for a verified patient.", 
                "parameters": { 
                    "type": "object", 
                    "properties": { 
                        "patient_id": {
                            "type": "string"
                        } 
                    }, 
                    "required": ["patient_id"], 
                    "additionalProperties": False 
                }, 
            }, 
        ), 
        ChatCompletionToolParam( 
            type="function", 
            function={ 
                "name": "create_refill_invoice", 
                "description": "Create a refill invoice for a selected medication and update patients_data.json.", 
                "parameters": { 
                    "type": "object", 
                    "properties": { 
                        "patient_id": {
                            "type": "string"
                        }, 
                        "medication_id": {
                            "type": "string"
                        }, 
                        "quantity": {
                            "type": "integer", 
                            "minimum": 1
                        }, 
                        "pharmacy": {
                            "type": "string"
                        } 
                    }, 
                    "required": ["patient_id", "medication_id", "quantity"], 
                    "additionalProperties": False 
                }, 
            }, 
        ), 
    ] 

    # llm = OpenAILLMService(
    #     model=os.getenv("OPENAI_MODEL"),
    #     api_key=os.getenv("OPENAI_API_KEY"),
    #     base_url=os.getenv("OPENAI_BASE_URL"),

    #     params=OpenAILLMService.InputParams(
    #         temperature=0.7,
    #         tools=tools_schema_,
    #         tool_choice="auto"
    #     )
    # )

    llm = OpenAILLMService(
        model=os.getenv("OPENAI_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        # base_url=os.getenv("OPENAI_BASE_URL")

        # params=OpenAILLMService.InputParams(
        #     temperature=0.7,
        #     tools=tools_schema_,
        #     tool_choice="auto"
        # )
    )

    messages = [
        {
            "role": "system",
            "content": agent_prompts.refill_prompt,
        }
    ]

    PATIENTS_PATH = "./patient_data.json"
    SESSIONS_DIR  = "./all_sessions"
    INVOICES_PATH = "./invoices.json"

    def _load_json(path, default):
        if not os.path.exists(path): 
            return default
        with open(path) as f: 
            try: 
                return json.load(f)
            except json.JSONDecodeError: 
                return default

    def _save_json(path, obj):
        with open(path, "w") as f: 
            json.dump(obj, f, indent=2)


    async def handle_confirm_identity(params: FunctionCallParams):
        args = params.arguments
        pid, fn = args["patient_id"].strip(), args["first_name"].strip()
        patients = _load_json(PATIENTS_PATH, {})
        rec = patients.get(pid)
        verified = bool(rec and rec["first_name"].lower()==fn.lower())

        # session_record = {
        #     "session_id": str(uuid.uuid4()),
        #     "ts": int(time.time()),
        #     "patient_id": pid,
        #     "verified": verified
        # }
        # _save_json(SESSION_PATH, session_record)

        # Return minimal info to the model (no extra PHI)
        result = {
            "verified": verified,
            "patient_id": pid,
            "display_name": f"{rec['first_name']} {rec['last_name']}" if rec else None
        }
        await params.result_callback(result)

    async def handle_list_active_meds(params: FunctionCallParams):
        pid = params.arguments["patient_id"].strip()
        patients = _load_json(PATIENTS_PATH, {})
        print(f"\n\nPRINTING LIST OF PATIENTS:\n\n{patients}\n")
        rec = patients.get(pid)
        meds = rec["meds"] if rec else []
        # Return compact listing
        print(f"\n\nMEDS: {meds}\n\n")
        await params.result_callback({"patient_id": pid, "medications": meds})

    async def handle_create_refill_invoice(params: FunctionCallParams):
        pid = params.arguments["patient_id"].strip()
        med_id = params.arguments["medication_id"].strip()
        qty = int(params.arguments["quantity"])
        pharmacy = params.arguments.get("pharmacy")

        invoice = {
            "invoice_id": f"INV-{uuid.uuid4().hex[:8].upper()}",
            "patient_id": pid,
            "medication_id": med_id,
            "quantity": qty,
            "pharmacy": pharmacy,
            "amount": round(15.0 + 2.0*qty, 2),  # dummy calc
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        invoices = _load_json(INVOICES_PATH, [])
        invoices.append(invoice)
        _save_json(INVOICES_PATH, invoices)
        await params.result_callback(invoice)

    llm.register_function("confirm_identity", handle_confirm_identity)
    llm.register_function("list_active_medications", handle_list_active_meds)
    llm.register_function("create_refill_invoice", handle_create_refill_invoice)

    context = OpenAILLMContext(messages, tools_schema_)
    context_aggregator = llm.create_context_aggregator(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,  # RTVI processor
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append({"role": "user", "content": "Say hello, and introduce yourself as a MEDICATION REFILL agent for a healthcare practice, and ask for their identity'"})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
