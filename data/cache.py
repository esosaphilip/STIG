
import redis


class StigCache:
    """
    This is to store and retieve already answered Questions.
    """

    def __init__(self, host="localhost", port=6379, ttl=86400, db=0):
        # ttl = time to live in seconds
        # 86400 = 24 hours — why did I choose this default?
        # connect to redis here
        self.ttl = ttl
        self.client = redis.Redis(host=host, port=port, db=db)
        
        pass

    def get(self, question: str) -> str | None:
        retrieved_answer = self.client.get(question)
        if retrieved_answer is None:
            return None
        if isinstance(retrieved_answer, bytes):
            return retrieved_answer.decode('utf-8')
        return str(retrieved_answer)

    def set(self, question: str, answer: str) -> None:
        if answer:
            self.client.set(question, answer, ex=self.ttl)