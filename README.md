# Indoor Positioning System using GAN
# or
# Indoor Positioning System for finding the location and predicting missing Fingerprints

This project is titled **Indoor Positioning System using GAN** and was developed by **Nooruddin Noonari** as part of the Master's in Software Engineering program at **Tianjin University, Tianjin, China**.

## Project Overview

The Indoor Positioning System (IPS) leverages a Generative Adversarial Network (GAN) to improve the accuracy and robustness of location-based services within indoor environments. Traditional WiFi-based positioning systems often struggle with issues like signal fluctuation and the high cost of building extensive fingerprint databases. This project addresses these challenges by using GANs to generate synthetic WiFi fingerprints, which enhance the system’s ability to predict positions accurately even in challenging environments.

## Features

- **Synthetic Fingerprint Generation:** The system uses a GAN to generate synthetic WiFi fingerprints, augmenting the existing dataset and reducing the need for extensive manual data collection.
- **Improved Accuracy:** By leveraging synthetic data, the IPS can provide more accurate location predictions, even in environments with sparse data or high variability in signal strength.
- **Multi-frequency Support:** The system supports both 2.4GHz and 5GHz WiFi signals, optimizing for range and interference resistance.
- **Scalable Implementation:** The system is designed to be scalable, allowing for easy expansion to cover larger areas or integrate with additional sensors.

## Project Scope

The project focuses on enhancing indoor positioning accuracy within medium to large indoor environments, such as university campuses, shopping malls, or office buildings. While the current implementation is based on WiFi signals, the framework is extensible to include other types of signals or sensors in future iterations.

## Technical Approach

1. **WiFi Signal Collection:** RSSI values are collected from multiple access points at various locations.
2. **Data Normalization:** The collected data is preprocessed and normalized to serve as input for the GAN model.
3. **GAN Model:** The GAN consists of a generator that creates synthetic RSSI values and a discriminator that evaluates the authenticity of the generated data.
4. **Model Training:** The GAN is trained using real RSSI data, and the generator learns to produce realistic fingerprints that can be used to augment the original dataset.

### Simplified Workflow

- The system collects WiFi signal strengths from various access points within an indoor environment.
- The collected data is normalized and used to train the GAN model.
- The GAN generates synthetic data to fill gaps in the original dataset.
- The augmented dataset is used to improve the accuracy of the positioning system.

## System Flow

![image](https://github.com/user-attachments/assets/6a3de19e-4bb7-4778-9aa0-fbc5515bbaba)

## GAN Model Architecture

![image](https://github.com/user-attachments/assets/694c24ac-75e0-4050-89fb-e6de82721f73)

## How to Run the Project

1. **Prerequisites:**
   - Python 3.x
   - TensorFlow or PyTorch (for GAN implementation)
   - Pandas, NumPy, and other necessary libraries

2. **Steps:**
   - Clone the repository to your local machine.
   - Install the required Python packages using `pip install -r requirements.txt`.
   - Run the data preprocessing script to normalize the RSSI data.
   - Train the GAN model using the training script.
   - Use the trained GAN to generate synthetic data and improve the indoor positioning system.

## Results 1

![Figure_1](https://github.com/user-attachments/assets/c548c007-a4ef-4d0a-af33-d9a2728d6ab4)


## Results 3

![Figure_2](https://github.com/user-attachments/assets/8e73ca16-3bb8-4e09-aefc-189bc2510f38)

## Future Work

- **Signal Fusion:** Integrating additional types of signals, such as Bluetooth or UWB, to enhance positioning accuracy.
- **Real-Time Positioning:** Developing a real-time positioning system that continuously updates as new data is collected.
- **Cross-Building Support:** Expanding the system to support multi-building environments, allowing seamless indoor navigation across different structures.

## Author

**Nooruddin Noonari**

- Email: noor.cs2@yahoo.com
- LinkedIn: [https://www.linkedin.com/in/noonari/]
- GitHub: [https://github.com/noorcs39]

This project was developed as part of my Master's in Software Engineering at Tianjin University, Tianjin, China.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
