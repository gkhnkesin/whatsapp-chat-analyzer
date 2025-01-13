import re
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime
import emoji
from wordcloud import STOPWORDS
import nltk
from nltk.corpus import stopwords
import os
import json

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class WhatsAppChatAnalyzer:
    """
    A class to analyze WhatsApp chat data.

    Attributes:
        file_path (str): Path to the WhatsApp chat file.
        messages (list): Parsed chat messages.
        stop_word_list (list): List of stopwords for filtering messages.
    """

    def __init__(self, file_path, language="english"):
        """
        Initializes the WhatsAppChatAnalyzer with the specified chat file path.

        Args:
            file_path (str): Path to the WhatsApp chat file.
        """
        self.file_path = file_path
        self.messages = []
        self.language = language
        try:
            self.stop_word_list = stopwords.words(language)
        except OSError:
            print(f"Stopwords for language '{language}' not found. Using an empty stopword list.")
            self.stopwords = []

    def read_chat(self):
        """
        Reads the chat file and stores the raw data.

        Returns:
            list: Lines from the chat file.
        """
        with open(self.file_path, 'r', encoding='utf-8') as file:
            data = file.readlines()
        return data

    def parse_chat(self, data):
        """
        Parses the raw chat data into structured messages.

        Args:
            data (list): Raw chat lines.

        Returns:
            list: Parsed messages with timestamp, sender, and message.
        """
        for line in data:
            if "Messages and calls are end-to-end encrypted." in line:
                continue

            match = re.match(r"(\d{1,2}\.\d{1,2}\.\d{4} \d{1,2}:\d{2}) - (.*?): (.*)", line)
            if match:
                timestamp, sender, message = match.groups()
                timestamp = datetime.strptime(timestamp, "%d.%m.%Y %H:%M")
                self.messages.append({"timestamp": timestamp, "sender": sender, "message": message})

    def preprocess_message(self, message):
        """
        Preprocesses a message by converting to lowercase and removing stopwords.

        Args:
            message (str): The original message.

        Returns:
            str: Preprocessed message.
        """
        words = nltk.word_tokenize(message.lower())
        filtered_words = [word for word in words if word not in self.stop_word_list]
        return " ".join(filtered_words)

    def analyze_messages(self):
        """
        Analyzes the parsed messages and generates various insights.
        """
        # Create a DataFrame
        df = pd.DataFrame(self.messages)
        df["hour"] = df["timestamp"].dt.hour
        df["day"] = df["timestamp"].dt.day_name()
        df["date"] = df["timestamp"].dt.date

        df['message'] = df['message'].apply(self.preprocess_message)

        # Message counts by sender
        sender_counts = df["sender"].value_counts()
        print("Message Counts:")
        print(sender_counts)

        # Weekday vs. weekend analysis
        df["is_weekend"] = df["day"].isin(["Saturday", "Sunday"])
        print("\nWeekday vs Weekend Message Distribution:")
        print(df.groupby("is_weekend")["message"].count())

        # Average words per message
        df["word_count"] = df["message"].apply(lambda x: len(x.split()))
        avg_word_count = df.groupby("sender")["word_count"].mean()
        print("\nAverage Words Per Message:")
        print(avg_word_count)

        # Longest and shortest messages
        longest_message = df.loc[df["word_count"].idxmax()]
        shortest_message = df.loc[df["word_count"].idxmin()]
        print("\nLongest Message:")
        print(longest_message)
        print("\nShortest Message:")
        print(shortest_message)

        # Generate word cloud
        self.generate_word_cloud(" ".join(df["message"]))

        # Hourly message distribution
        self.plot_hourly_distribution(df)

        # Emoji analysis
        self.analyze_emojis(df)

        # Response times
        self.analyze_response_times(df)


        # Export analysis results
        self.export_results(df)

    def generate_word_cloud(self, text):
        """
        Generates and displays a word cloud.

        Args:
            text (str): The combined text for the word cloud.
        """
        wordcloud = WordCloud(
            width=800, height=400, background_color="white",
            stopwords=STOPWORDS.union(set(["bir", "çok", "ama", "yine", "diye"]))
        ).generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud")
        plt.show()

    def plot_hourly_distribution(self, df):
        """
        Plots the hourly distribution of messages.

        Args:
            df (pd.DataFrame): DataFrame containing message data.
        """
        hourly_counts = df.groupby("hour")["message"].count()
        plt.figure(figsize=(10, 5))
        hourly_counts.plot(kind="bar", color="orange")
        plt.title("Hourly Message Distribution")
        plt.xlabel("Hour")
        plt.ylabel("Message Count")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def analyze_emojis(self, df):
        """
        Analyzes emoji usage in messages.

        Args:
            df (pd.DataFrame): DataFrame containing message data.
        """
        df["emojis"] = df["message"].apply(lambda x: ''.join(c for c in x if c in emoji.EMOJI_DATA))
        emoji_counts = Counter("".join(df["emojis"]))
        print("\nMost Common Emojis:")
        print(emoji_counts.most_common(10))

    def analyze_response_times(self, df):
        """
        Analyzes response times between messages.

        Args:
            df (pd.DataFrame): DataFrame containing message data.
        """
        df["response_time"] = df["timestamp"].diff().dt.total_seconds().fillna(0)
        avg_response_time = df[df["response_time"] > 0]["response_time"].mean()
        print(f"\nAverage Response Time: {avg_response_time:.2f} seconds")


    def export_results(self, df):
        """
        Exports analysis results to files.

        Args:
            df (pd.DataFrame): DataFrame containing message data.
        """
        output_dir = "analysis_results"
        os.makedirs(output_dir, exist_ok=True)

        # Save DataFrame to CSV
        csv_path = os.path.join(output_dir, "chat_analysis.csv")
        df.to_csv(csv_path, index=False)

        # Save summary to JSON
        summary = {
            "message_counts": df["sender"].value_counts().to_dict(),
            "average_words_per_message": df.groupby("sender")["word_count"].mean().to_dict()
        }
        json_path = os.path.join(output_dir, "summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)

        print(f"Results exported to {output_dir}")

# Example usage
if __name__ == "__main__":
    file_path = "‎your-text.txt"  # Replace with your file path
    language = "turkish"
    analyzer = WhatsAppChatAnalyzer(file_path, language)
    chat_data = analyzer.read_chat()
    analyzer.parse_chat(chat_data)
    analyzer.analyze_messages()
