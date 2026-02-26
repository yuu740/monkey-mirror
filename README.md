# 🐒 Monkey Mirror Ultra Precision
**Monkey Mirror Ultra Precision** is an interactive AI-powered application that transforms your real-time facial expressions and hand gestures into legendary monkey memes. Unlike standard pose detection, this app utilizes high-precision landmarks to capture specific finger movements and lip distancing.


## ✨ Key Features
- _**Dual-View Interface**_: Displays the original webcam input with visualized landmarks on the left and the transformed monkey meme on the right.
- _**High-Precision Tracking**_: Leverages MediaPipe Face Mesh (468 points) and Hand Tracking (21 points) to detect index finger proximity and mouth opening ratios with extreme accuracy.
- _**Real-time Confidence Dashboard**_: Displays a live percentage for each state (AHA, THINKING, SCREAM) to show how closely your gesture matches the target.
- _**Dynamic Normalized Scaling**_: Uses the distance between eyes as a standard unit of measurement, ensuring detection remains accurate regardless of how close or far you are from the camera.


## 🛠️ Gesture Logic & Mathematics
The application calculates states based on spatial coordinates and the Euclidean distance formula:
$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$

**Gesture Triggers**:
- _**AHA!**_: Triggered when the index finger tip is above eye level and near the ear region (Left or Right).
- _**THINKING**_: Activated when the index finger tip is in close proximity to the nose or mouth center.
- _**SCREAM**_: Detected when the ratio of the distance between the upper and lower lips exceeds a specific threshold relative to eye distance.
- _**DEFAULT**_: The base state displayed when no other gestures reach the 35% confidence threshold.


## 📦 Installation & Setup
#### Prerequisites
- _**Python 3.11 or 3.12**_: Highly recommended over version 3.13 for MediaPipe stability on Windows.
- An active webcam.
#### Installation Steps
- Clone this repository to your local machine.
- Create and activate a Virtual Environment:
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
- Install the required dependencies:
  ```powershell
  pip install opencv-python mediapipe numpy
  ```
- Ensure your project structure matches the following:
  ```plaintext
  monkey-mirror/
  ├── assets/
  │   ├── monkey-default.jpg
  │   ├── monkey-scream.jpg
  │   ├── monkey-thinking.jpg
  │   └── monkey-aha.jpg
  └── monkey_mirror.py
  ```

  
## 🏗️ Building the Standalone Executable (.exe)
To share this app as a single file that doesn't require a Python installation, follow these steps:
- Install PyInstaller within your venv: `pip install pyinstaller`.
- Run the following build command:
  ```powershell
  pyinstaller --noconfirm --onefile --windowed `
  --add-data "assets;assets" `
  --collect-all mediapipe `
  monkey_mirror.py
  ```
  Note: The --collect-all mediapipe flag is mandatory to prevent FileNotFoundError in the Windows Temp folder by including    all necessary binary models.


##  💬 Feedback & Contributions
Feedback is highly appreciated! If you encounter issues with detection sensitivity or have ideas for new monkey states, feel free to Open an Issue or submit a Pull Request.

#### 🚩 How to Report a Bug or Request a Feature
If you find a bug (e.g., the "Aha" state is flickering) or have a cool idea (e.g., adding sound effects), please follow these steps:

- Navigate to the Issues tab at the top of this repository.

- Click the green New Issue button.

- Provide a clear title (e.g., [BUG] Scream triggers too easily).

- Describe the issue or feature in detail and click Submit new issue.

#### 🛠️ How to Submit a Pull Request
Want to improve the code or fix a bug yourself? Follow this standard open-source workflow:

- Fork the Project (Click the 'Fork' button at the top right of this page).

- Clone your fork to your local machine.

- Create your Feature Branch (`git checkout -b feature/AmazingFeature`).

- Commit your changes (`git commit -m 'Add some AmazingFeature'`).

- Push to the branch (`git push origin feature/AmazingFeature`).

- Open a Pull Request from your fork's page back to this original repository.
