# Smart Traffic Control with AdaptFlow-TSC

This project implements an advanced Federated Reinforcement Learning (FRL) system for Traffic Signal Control (TSC), featuring the novel **AdaptFlow** algorithm.

## 📖 Documentation

For detailed technical information, please refer to the following guides:

*   **[AdaptFlow Simulation Flow](docs/simulation_flow.md)**: An end-to-end breakdown of how the simulation works, from frontend triggers to spatio-temporal GAT agents.
*   **[AdaptFlow Novelties & Comparison](docs/adaptflow_novelties.md)**: A deep dive into the unique features of AdaptFlow and how it compares to standard algorithms like FedAvg and FedKD.

## 🚀 Getting Started

### 1. Prerequisites

Before setting up the project, ensure you have the following installed:
*   **Python 3.9+** (For the backend RL logic and server)
*   **Node.js 16+ & npm** (For the React frontend)
*   **SUMO (Simulation of Urban MObility)**: Required for high-fidelity traffic simulations.
    *   *Mac*: `brew install sumo`
    *   *Windows/Linux*: Follow the [SUMO Installation Guide](https://eclipse.org/sumo/intro/index.php#installation).

### 2. Installation

#### Backend Setup
1.  **Navigate to backend**: `cd backend`
2.  **Create Virtual Environment**: `python -m venv venv`
3.  **Activate Venv**:
    *   *Mac/Linux*: `source venv/bin/activate`
    *   *Windows*: `.\venv\Scripts\activate`
4.  **Install Dependencies**: `pip install -r requirements.txt`

#### Frontend Setup
1.  **Navigate to frontend**: `cd frontend`
2.  **Install Packages**: `npm install`

### 3. Environment Setup

**Backend (`backend/.env`):**
```env
TOMTOM_API_KEY=your_api_key_here
```

**Frontend (`frontend/.env`):**
```env
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
```

### 4. Running the Project

1.  **Backend**: Navigate to `backend/` and run `./venv/bin/python server.py`.
2.  **Frontend**: Navigate to `frontend/` and run `npm run dev`.
3.  **Simulation**: Use the dashboard to place pins, select "AdaptFlow", and hit simulation.
