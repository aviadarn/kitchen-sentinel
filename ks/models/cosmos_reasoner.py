"""NVIDIA's Cosmos3 Reasoner as a zero-shot accident anticipator: one yes/no question per window.

The Reasoner is the vision-language half of the Cosmos3 world model (a Qwen3-VL-8B architecture; the
Transformers integration loads only that tower, ~17 GB in bf16). Each 2 s window goes in at the model
card's recommended 4 fps with one fixed question, asked without the model's step-by-step reasoning,
and the score is P(yes) against P(no) for the first answer token. Like BADAS-Open, it is used as is:
nothing is tuned on our labels, and the question was fixed before any window was scored.
"""

import numpy as np
import torch

MODEL = "nvidia/Cosmos3-Nano"
FPS = 4.0  # model card: "Video inputs are recommended at a frame rate of 4 fps"
SYSTEM = "You are a helpful assistant."  # the prompt guide's system prompt
QUESTION = ("Is an accident about to happen or starting right now in this video, such as someone getting cut "
            "or burned, a fire, a person slipping or falling, or something falling, spilling or breaking? "
            "Answer yes or no.")
YES = ("Yes", "yes", " Yes", " yes", "YES")
NO = ("No", "no", " No", " no", "NO")


def subsample(windows, src_fps, fps=FPS):
    """Every (src_fps / fps)-th frame of (n, T, H, W, 3) windows, ending on the last (scored) frame."""
    step = int(round(src_fps / fps))
    return windows[:, step - 1::step]


def answer_probability(logits, yes_ids, no_ids):
    """From next-token logits (n, vocab): P(yes | the answer is yes or no), and the probability the
    model puts on those tokens at all (low = it wanted to say something else)."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    yes, no = torch.logsumexp(logp[:, yes_ids], dim=-1), torch.logsumexp(logp[:, no_ids], dim=-1)
    return torch.sigmoid(yes - no), yes.exp() + no.exp()


class CosmosReasoner:
    """windows (n, T, H, W, 3) uint8 at `src_fps` -> {"cosmos": P(yes), "cosmos_mass": yes+no probability}."""

    specs = {"cosmos": ((), np.float32), "cosmos_mass": ((), np.float32)}

    def __init__(self, device, batch=16, model_id=MODEL, src_fps=8.0, dtype=torch.bfloat16, model=None):
        from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration
        from transformers.video_utils import VideoMetadata

        self.device, self.batch, self.src_fps, self.dtype = device, batch, src_fps, dtype
        self.metadata = VideoMetadata
        self.processor = AutoProcessor.from_pretrained(model_id)
        model = model or Cosmos3OmniForConditionalGeneration.from_pretrained(model_id, dtype=dtype)
        self.model = model.to(device).eval()
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": QUESTION}]}]
        self.text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tok = self.processor.tokenizer
        ids = lambda words: [i for w in words for i in tok.encode(w, add_special_tokens=False)[:1]]
        self.yes_ids, self.no_ids = ids(YES), ids(NO)

    def inputs(self, clips):
        n, t = clips.shape[:2]
        meta = [self.metadata(total_num_frames=t, fps=FPS, frames_indices=list(range(t))) for _ in range(n)]
        batch = self.processor(text=[self.text] * n, videos=list(clips), video_metadata=meta, do_sample_frames=False,
                               cap_pixels_per_frame=True, padding=True, return_tensors="pt")
        batch = batch.to(self.device)
        batch["pixel_values_videos"] = batch["pixel_values_videos"].to(self.dtype)
        return batch

    @torch.no_grad()
    def __call__(self, windows):
        clips = subsample(windows, self.src_fps).cpu()
        p_yes, mass = [], []
        for i in range(0, len(clips), self.batch):
            batch = self.inputs(clips[i:i + self.batch])
            # Every window has the same prompt and frame size, so no row is padded: the last position
            # is each row's next token.
            logits = self.model(**batch, logits_to_keep=1).logits[:, -1]
            p, m = answer_probability(logits, self.yes_ids, self.no_ids)
            p_yes.append(p.cpu())
            mass.append(m.cpu())
        return {"cosmos": torch.cat(p_yes).numpy(), "cosmos_mass": torch.cat(mass).numpy()}
