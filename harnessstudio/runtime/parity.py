"""Reject native/null drift before physical actions; no evaluator outcomes used."""
from dataclasses import dataclass
import re


class ParityError(RuntimeError):
    """Infrastructure mismatch, never a scored robot-policy failure."""


@dataclass(frozen=True)
class ReferenceDecision:
    step: int
    public_input_sha256: str
    instruction_sha256: str
    action_sha256: str
    execute_steps: int


@dataclass(frozen=True, init=False)
class ParityGuard:
    decisions: tuple[ReferenceDecision, ...]

    def __init__(self, baseline_traces):
        decisions=[]
        for row in baseline_traces:
            if row.get('arm')!='baseline' or row.get('policy_calls')!=1:
                raise ValueError('Parity requires actual one-proposal native baseline traces')
            actions=row['proposal_action_sha256']
            if len(actions)!=1 or row.get('selected_index')!=0:
                raise ValueError('Baseline trace must execute its one original proposal')
            record=ReferenceDecision(row['step'],row['public_input_sha256'],row['instruction_sha256'],
                actions[0],row['execute_steps'])
            if (type(record.step) is not int or record.step<0 or type(record.execute_steps) is not int
                    or record.execute_steps<=0 or any(not isinstance(h,str) or not re.fullmatch(r'[0-9a-f]{64}',h)
                    for h in (record.public_input_sha256,record.instruction_sha256,record.action_sha256))):
                raise ValueError('Malformed native baseline trace')
            decisions.append(record)
        if not decisions or decisions[0].step!=0 or any(
                b.step!=a.step+a.execute_steps for a,b in zip(decisions,decisions[1:])):
            raise ValueError('Baseline traces must form a contiguous native execution sequence')
        object.__setattr__(self,'decisions',tuple(decisions))

    def expected(self, step):
        for row in self.decisions:
            if row.step==step:return row
        raise ParityError('No native baseline evidence for this decision; stop before action')

    def before_inference(self, step, public_input_sha256, instruction_sha256, execute_steps):
        row=self.expected(step)
        if (row.public_input_sha256,row.instruction_sha256,row.execute_steps)!=(
                public_input_sha256,instruction_sha256,execute_steps):
            raise ParityError('Native/null input or cadence drift; stop before inference/action')

    def before_action(self, step, action_sha256):
        if self.expected(step).action_sha256!=action_sha256:
            raise ParityError('Native/null action drift; stop before physical action')
