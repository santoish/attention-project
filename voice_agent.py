"""
voice_agent.py - voice assistant for the Classroom Attention app (LiveKit Agents)
---------------------------------------------------------------------------------
Run (from the project folder, in the voice environment):
    python voice_agent.py console     talk using this computer's mic and speaker
    python voice_agent.py dev         connect to LiveKit Cloud (use the Agent Console,
                                      agent name: attention-assistant)

The attention app (attention_app.py) must be running in its own terminal. It writes
live_status.json every second; this assistant reads it through status_bridge.py.
Only numbers are shared with the assistant, never video or images.
"""
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import (Agent, AgentServer, AgentSession, RunContext,
                            TurnHandlingOptions, function_tool, inference)

import status_bridge as bridge

load_dotenv(".env.local")

INSTRUCTIONS = """You are a voice assistant for a teacher who uses a classroom
attention monitoring app. The app watches a camera and scores how attentive each
student is from 0 to 100, where 100 is fully attentive and 0 is drowsy.
Always use your tools to get live information. Never guess or invent numbers.
Refer to students only by their number, never by name.
Keep answers short and natural for speech, without lists, symbols or emojis.
If the app is not running, tell the teacher to start the webcam or open a video.
Remind the teacher, if asked, that the scores are estimates from head direction
and eye closure, so looking down to write notes can look like inattention."""


class AttentionAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS)

    @function_tool
    async def get_class_attention(self, context: RunContext) -> str:
        """Get the live attention level of the whole class: the average score and
        how many students are attentive, not attentive, or drowsy."""
        return bridge.describe_class(bridge.read_status())

    @function_tool
    async def get_student_status(self, context: RunContext, student_id: int) -> str:
        """Get the live attention status and score of one student.

        Args:
            student_id: The student's number shown on the video, for example 2.
        """
        return bridge.describe_student(bridge.read_status(), student_id)

    @function_tool
    async def get_recent_alerts(self, context: RunContext) -> str:
        """Get the most recent alerts, such as students who became distracted or
        drowsy, newest first."""
        return bridge.describe_alerts(bridge.read_status())

    @function_tool
    async def calibrate_students(self, context: RunContext) -> str:
        """Calibrate the app so each student's current head direction counts as
        looking at the board. Use when the teacher asks to calibrate or reset."""
        return bridge.request_calibration()


server = AgentServer()


@server.rtc_session(agent_name="attention-assistant")
async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=inference.STT(model="assemblyai/universal-3-5-pro", language="en"),
        llm=inference.LLM(model="google/gemma-4-31b-it"),
        tts=inference.TTS(
            model="fishaudio/s2.1-pro",
            voice="fa4c9eb3dccc4806b382b40d61c6b10a",
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
        ),
    )

    await session.start(room=ctx.room, agent=AttentionAssistant())

    await session.generate_reply(
        instructions="Greet the teacher briefly and say you can report on class "
                     "attention, individual students, and recent alerts."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
