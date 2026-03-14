import asyncio
from typing import Dict, List, Callable, Any
import json

class MessageBroker:
    """
    A simple asynchronous Pub/Sub broker for Federated Learning.
    Allows decoupling of Servers and Nodes.
    """
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.history: List[Dict] = []

    def subscribe(self, topic: str, callback: Callable):
        """Register a callback for a specific topic."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
        print(f"[Broker] New subscriber for topic: {topic}")

    async def publish(self, topic: str, message: Any):
        """Publish a message to all subscribers of a topic."""
        # Log to history for debugging/viz
        self.history.append({
            "topic": topic,
            "message_type": type(message).__name__,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        if topic in self.subscribers:
            tasks = [cb(message) for cb in self.subscribers[topic]]
            if tasks:
                await asyncio.gather(*tasks)
        
    def get_logs(self):
        return self.history

# Global broker instance
broker = MessageBroker()
