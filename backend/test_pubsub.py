import asyncio
import sys
import os

# Add parent dir to path
sys.path.append(os.getcwd())

from federated_learning.message_broker import broker

async def test_pubsub():
    print("Testing PUB-SUB Broker...")
    
    received_messages = []
    
    async def sample_callback(msg):
        print(f"  [Subscriber] Received: {msg}")
        received_messages.append(msg)
        
    broker.subscribe("test/topic", sample_callback)
    
    print("  [Publisher] Sending message...")
    await broker.publish("test/topic", {"status": "ok", "payload": "Hello FL"})
    
    if len(received_messages) > 0:
        print("SUCCESS: Message received via PUB-SUB!")
    else:
        print("FAILED: No message received.")
        exit(1)

if __name__ == "__main__":
    asyncio.run(test_pubsub())
