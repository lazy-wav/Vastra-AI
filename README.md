# Vastra AI

<div align="center"> <img src="https://github.com/user-attachments/assets/a22606be-1d01-4cac-a2dd-e6abe62144b8" alt="Logo"></div>

## Inspiration

80% of us only wear 20% of what's in our wardrobe. Vastra AI was inspired by the growing need to streamline personal fashion choices and maximize the use of existing wardrobe items. Observing how many people struggle to efficiently manage and style their clothing, this project harnesses artificial intelligence to provide smart, personalized fashion recommendations. Our goal is to help users make the most of their wardrobe by suggesting new outfit combinations and providing style guidance based on their existing clothing.


## What It Does

Vastra AI provides intelligent style suggestions by analyzing images of clothing items from the user’s wardrobe. The application uses advanced machine learning algorithms to understand the style, color, and type of clothing items. It then generates personalized recommendations for new outfit combinations, helping users discover fresh ways to wear the clothes they already own. This approach enhances wardrobe management and promotes sustainable fashion practices by encouraging the reuse of existing clothing.


## How We Built It

- **Languages:** Python was used for the implementation of machine learning models and overall application logic.
- **Frameworks:** TensorFlow and Keras were employed to build and train the machine learning models, utilizing deep learning techniques to analyze clothing images.
- **Models:** We used Convolutional Neural Networks (CNNs) and specifically fine-tuned ResNet50, a pre-trained deep learning model, to recognize and compare clothing items. This model was central to generating style recommendations.
- **Platforms:** Google Colab provided the necessary computational resources for training our models, offering access to GPUs for efficient processing.


## Challenges We Ran Into

- **Data Quality:** Ensuring that the training data was diverse and high-quality was a primary challenge. Gathering and accurately labeling a comprehensive dataset of clothing images was crucial for effective model training.
- **Model Training:** Training the Siamese network with ResNet50 involved numerous experiments with hyperparameters and fine-tuning to achieve the desired accuracy and performance.


## Accomplishments That We're Proud Of

- **Effective Model Training:** Successfully developed a machine learning model that provides accurate and useful fashion recommendations based on users' existing wardrobe.
- **Intuitive Interface:** Created a user-friendly interface that effectively communicates personalized outfit suggestions and enhances user engagement.



## What We Learned

- **Advanced Techniques:** Gained valuable experience in using TensorFlow and Keras for developing and deploying deep learning models.
- **Data Management:** Understood the critical role of high-quality data and preprocessing in achieving model accuracy and reliability.


## What's Next for Vastra AI

- **Model Enhancement:** Plan to incorporate advanced data augmentation techniques and further fine-tuning to boost model performance and accuracy.
- **Feature Expansion:** Explore additional features such as virtual try-ons and trend analysis to enrich the user experience and offer more comprehensive style recommendations.
- **User Feedback:** Collect and analyze feedback from users to refine and enhance the application based on real-world use and preferences.
- **Deployment and Scalability:** Focus on preparing the application for broader deployment, ensuring it can scale effectively to accommodate a larger user base and diverse wardrobe collections.



## Getting Started

To get a local copy of the project up and running, follow these simple steps:

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/vastra-ai.git
    ```

2. **Navigate to the project directory:**

    ```sh
    cd vastra-ai
    ```

3. **Install the required dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Run the application:**

    ```sh
    streamlit run app.py
    ```

## Contributing

Contributions are welcome! If you have suggestions or improvements, please fork the repository and submit a pull request. For more details, check out our [contributing guidelines](CONTRIBUTING.md).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
