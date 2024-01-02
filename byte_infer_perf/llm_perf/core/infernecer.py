from typing import Any, Dict, Iterable

from llm_perf.core.common import GenerateConfig, GenerateRequest
from llm_perf.core.scheduler import CoreScheduler
from llm_perf.utils.logger import logger


class CoreInferencer:
    def __init__(self, scheduler: CoreScheduler) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.tokenizer = scheduler.tokenizer
        self.add_sep_token = scheduler.add_sep_token

        self.scheduler.start()
        self.warmup()

    def prepare_request(
        self, prompt: str, generate_config: Dict[str, Any]
    ) -> GenerateRequest:
        input_ids = self.tokenizer.encode(prompt)
        if self.add_sep_token:
            input_ids.append(self.tokenizer.sep_token_id)
        config = GenerateConfig(
            min_new_tokens=generate_config.get("min_new_tokens", 0),
            max_new_tokens=generate_config.get("max_new_tokens", 0),
            top_k=generate_config.get("top_k", 0),
            top_p=generate_config.get("top_p", 1.0),
            temperature=generate_config.get("temperature", 1.0),
            presence_penalty=generate_config.get("presence_penalty", 1.0),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            get_input_logits=bool(generate_config.get("get_input_logits", 0)),
        )
        req = GenerateRequest(input_ids, config)
        return req

    def streaming_inference(
        self, prompt: str, generate_config: Dict[str, Any]
    ) -> Iterable[Dict[str, Any]]:
        req = self.prepare_request(prompt, generate_config)
        for gen_res in self.scheduler.generate(req):
            if req.generate_config.get_input_logits:
                res, ppl, dump_file = gen_res
                result = {"ppl": ppl, "dump_file": dump_file}
            else:
                res = gen_res
                result = {"ppl": 0, "dump_file": ""}
            token = self.tokenizer.decode(res.token_id)
            result["token"] = token
            logger.debug(f"inference result: {result}")
            yield result

    def warmup(self):
        prompt = "1+1=?"
        # prompt = "你好"
        #         prompt = "有关涉外仲裁协议的效力问题，下列表述不正确的是：____ \
        # A. 约定的仲裁事项超出法律规定的范围的，仲裁协议无效 \
        # B. 如果仲裁协议对仲裁事项和仲裁委员会约定不明确，当事人不能达成补充协议的，该仲裁协议无效 \
        # C. 当事人约定两个或两个以上的仲裁机构进行仲裁的，该仲裁协议无效 \
        # D. 当事人达成的仲裁协议只规定了仲裁地点，未约定仲裁机构，双方当事人在补充协议中选定了在该地点依法重新组建的仲裁机构的，仲裁协议有效 \
        # 答案："
        # prompt = "写一个美好结局的《卖火柴的小女孩》童话故事"
        generate_config = {
            "min_new_tokens": 1,
            "max_new_tokens": 512,
            "top_k": 1,
            "temperature": 0.2,
            "presence_penalty": 1.0,
        }
        logger.info(f"warmup prompt: {prompt}\nconfig: {generate_config}")

        res = ""
        for result in self.streaming_inference(prompt, generate_config):
            res += result["token"]

        logger.info(f"warmup response:{res}")
