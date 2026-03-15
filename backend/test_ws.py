import asyncio
import json
import websockets

async def test_ws():
    uri = "ws://localhost:8000/api/simulate"
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected.")
            req = {
                "city": "Delhi",
                "algorithm": "AdaptFlow",
                "use_tomtom": True,
                "target_pois": ["healthcare"]
            }
            print(f"Sending request: {req}")
            await websocket.send(json.dumps(req))
            
            while True:
                response = await websocket.recv()
                print(f"Received: {response}")
    except websockets.ConnectionClosed as e:
        print(f"Connection closed: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_ws())
