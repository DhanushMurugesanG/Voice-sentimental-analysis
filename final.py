import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import librosa
import glob
import json
import base64
import hashlib
import soundfile as sf
import sounddevice as sd
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import logging
import io
from typing import Optional, Tuple, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioProcessingError(Exception):
    pass

class BlockchainError(Exception):
    pass

class ModelError(Exception):
    pass

# Feature Extraction Functions
def extract_feature(file_name=None, audio=None, sample_rate=22050, mfcc=True, chroma=True, mel=True):
    if file_name:
        try:
            with sf.SoundFile(file_name) as sound_file:
                X = sound_file.read(dtype="float32")
                sample_rate = sound_file.samplerate
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            raise AudioProcessingError(f"Failed to load audio file: {e}")
    else:
        X = audio

    result = np.array([])

    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))

    if chroma:
        stft = np.abs(librosa.stft(X))
        chroma_features = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_features))

    if mel:
        mel_features = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel_features))

    return result

# Emotion Recognition Model
class EmotionRecognizer:
    def __init__(self, model_path: Optional[str] = None):
        self.emotions = ['calm', 'happy', 'fearful', 'disgust']
        self.height = 15
        self.width = 12
        self.channels = 1
        self.model = None
        self.label_encoder = None
        self.scaler = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.train_model()

    def train_model(self):
        logger.info("Training new emotion recognition model...")
        try:
            # Load and preprocess data
            x, y = self._load_training_data()
            
            # Split and normalize data
            (x_train, x_test, y_train, y_test), self.label_encoder = self._prepare_data(x, y)
            
            # Create and train model
            self.model = self._create_model()
            self.model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))
            
            logger.info("Model training completed successfully")
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise ModelError(f"Failed to train model: {e}")

    def _load_training_data(self):
        x, y = [], []
        emotions = {
            '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
        }
        
        dataset_path = os.path.join(os.getcwd(), "speech-emotion-recognition-ravdess-data/Actor_*/*.wav")
        print(f"Looking for audio files in: {dataset_path}")
        for file in glob.glob(dataset_path):
            file_name = os.path.basename(file)
            emotion = emotions.get(file_name.split("-")[2])
            
            if emotion not in self.emotions:
                continue
                
            feature = extract_feature(file_name=file)
            x.append(feature)
            y.append(emotion)
            
        return np.array(x), y

    def _prepare_data(self, x, y):
        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        
        # Split data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9)
        
        # Scale features
        self.scaler = StandardScaler()
        x_train = self.scaler.fit_transform(x_train)
        x_test = self.scaler.transform(x_test)
        
        # Reshape for CNN
        x_train = x_train.reshape(-1, self.height, self.width, self.channels)
        x_test = x_test.reshape(-1, self.height, self.width, self.channels)
        
        return (x_train, x_test, y_train, y_test), label_encoder

    def _create_model(self):
        model = Sequential([
            tf.keras.layers.Input(shape=(self.height, self.width, self.channels)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(len(self.emotions), activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model

    def predict_emotion(self, audio_data: np.ndarray, sample_rate: int) -> str:
        try:
            # Extract features
            features = extract_feature(audio=audio_data, sample_rate=sample_rate)
            features = self.scaler.transform([features])
            features = features.reshape(-1, self.height, self.width, self.channels)
            
            # Predict
            prediction = np.argmax(self.model.predict(features), axis=1)
            predicted_emotion = self.label_encoder.inverse_transform(prediction)[0]
            
            return predicted_emotion
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ModelError(f"Failed to predict emotion: {e}")

    def save_model(self, path: str):
        try:
            self.model.save(path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise ModelError(f"Failed to save model: {e}")

    def load_model(self, path: str):
        try:
            self.model = tf.keras.models.load_model(path)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelError(f"Failed to load model: {e}")

class Block:
    def __init__(self, index: int, timestamp: str, audio_data: str, emotion: str, previous_hash: str):
        self.index = index
        self.timestamp = timestamp
        self.audio_data = audio_data
        self.emotion = emotion
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        try:
            block_string = json.dumps({
                "index": self.index,
                "timestamp": self.timestamp,
                "audio_data": str(self.audio_data),
                "emotion": self.emotion,
                "previous_hash": self.previous_hash,
                "nonce": self.nonce
            }, sort_keys=True)
            return hashlib.sha256(block_string.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Hash calculation failed: {e}")
            raise BlockchainError("Failed to calculate block hash")

    def mine_block(self, difficulty: int) -> None:
        try:
            while self.hash[:difficulty] != '0' * difficulty:
                self.nonce += 1
                self.hash = self.calculate_hash()
        except Exception as e:
            logger.error(f"Block mining failed: {e}")
            raise BlockchainError("Failed to mine block")

class EmotionAudioBlockchain:
    def __init__(self, difficulty: int = 2):
        self.chain = []
        self.difficulty = difficulty
        self.create_genesis_block()

    def create_genesis_block(self) -> None:
        try:
            genesis_block = Block(0, str(datetime.now()), "Genesis Block", "None", "0")
            genesis_block.mine_block(self.difficulty)
            self.chain.append(genesis_block)
        except Exception as e:
            logger.error(f"Genesis block creation failed: {e}")
            raise BlockchainError("Failed to create genesis block")

    def add_block(self, audio_data: str, emotion: str) -> Block:
        try:
            new_block = Block(
                len(self.chain),
                str(datetime.now()),
                audio_data,
                emotion,
                self.chain[-1].hash
            )
            new_block.mine_block(self.difficulty)
            self.chain.append(new_block)
            return new_block
        except Exception as e:
            logger.error(f"Failed to add block: {e}")
            raise BlockchainError("Failed to add new block")

    def is_chain_valid(self) -> bool:
        try:
            for i in range(1, len(self.chain)):
                current_block = self.chain[i]
                previous_block = self.chain[i-1]

                if current_block.hash != current_block.calculate_hash():
                    return False

                if current_block.previous_hash != previous_block.hash:
                    return False

            return True
        except Exception as e:
            logger.error(f"Chain validation failed: {e}")
            return False

    def get_statistics(self) -> dict:
        try:
            stats = {
                'total_recordings': len(self.chain) - 1,  # Exclude genesis block
                'emotions': {}
            }
            
            for block in self.chain[1:]:  # Skip genesis block
                if block.emotion in stats['emotions']:
                    stats['emotions'][block.emotion] += 1
                else:
                    stats['emotions'][block.emotion] = 1
                    
            return stats
        except Exception as e:
            logger.error(f"Failed to generate statistics: {e}")
            raise BlockchainError("Failed to generate statistics")

    def save_to_file(self, filename: str = "blockchain_data.json") -> None:
        try:
            blockchain_data = []
            for block in self.chain:
                block_data = {
                    'index': block.index,
                    'timestamp': block.timestamp,
                    'audio_data': block.audio_data,
                    'emotion': block.emotion,
                    'previous_hash': block.previous_hash,
                    'hash': block.hash,
                    'nonce': block.nonce
                }
                blockchain_data.append(block_data)

            with open(filename, 'w') as f:
                json.dump(blockchain_data, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save blockchain: {e}")
            raise BlockchainError("Failed to save blockchain to file")

    def load_from_file(self, filename: str = "blockchain_data.json") -> None:
        try:
            if not os.path.exists(filename):
                logger.info(f"No blockchain file found at {filename}")
                return

            with open(filename, 'r') as f:
                blockchain_data = json.load(f)

            self.chain = []
            for block_data in blockchain_data:
                block = Block(
                    block_data['index'],
                    block_data['timestamp'],
                    block_data['audio_data'],
                    block_data['emotion'],
                    block_data['previous_hash']
                )
                block.hash = block_data['hash']
                block.nonce = block_data['nonce']
                self.chain.append(block)
        except Exception as e:
            logger.error(f"Failed to load blockchain: {e}")
            raise BlockchainError("Failed to load blockchain from file")


def encode_audio(audio_data: np.ndarray, sample_rate: int) -> str:
    try:
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV')
        audio_bytes = buffer.getvalue()
        return base64.b64encode(audio_bytes).decode('utf-8')
    except Exception as e:
        logger.error(f"Audio encoding failed: {e}")
        raise AudioProcessingError("Failed to encode audio data")

def decode_audio(audio_base64: str) -> Tuple[np.ndarray, int]:
    try:
        audio_bytes = base64.b64decode(audio_base64)
        buffer = io.BytesIO(audio_bytes)
        audio_data, sample_rate = sf.read(buffer)
        return audio_data, sample_rate
    except Exception as e:
        logger.error(f"Audio decoding failed: {e}")
        raise AudioProcessingError("Failed to decode audio data")

def record_and_process(blockchain: EmotionAudioBlockchain, 
                      emotion_recognizer: EmotionRecognizer,
                      duration: int = 5, 
                      sample_rate: int = 22050) -> None:
    try:
        print(f"\nRecording for {duration} seconds...")
        audio = sd.rec(int(duration * sample_rate), 
                      samplerate=sample_rate, 
                      channels=1, 
                      dtype='float32')
        sd.wait()
        audio = audio.flatten()

        # Predict emotion
        emotion = emotion_recognizer.predict_emotion(audio, sample_rate)
        
        # Encode and store in blockchain
        audio_base64 = encode_audio(audio, sample_rate)
        block = blockchain.add_block(audio_base64, emotion)
        
        print(f"\nPredicted Emotion: {emotion}")
        print(f"Block Hash: {block.hash}")
        
        # Save blockchain
        blockchain.save_to_file()
        
    except Exception as e:
        logger.error(f"Recording and processing failed: {e}")
        print(f"An error occurred: {str(e)}")
def play_audio_from_block(block: Block) -> None:
    try:
        if block.audio_data == "Genesis Block":
            print("Cannot play audio from genesis block")
            return
            
        audio_data, sample_rate = decode_audio(block.audio_data)
        sd.play(audio_data, sample_rate)
        sd.wait()
        
    except Exception as e:
        logger.error(f"Failed to play audio: {e}")
        print(f"An error occurred while playing audio: {str(e)}")

def main():
    try:
        print("\nInitializing Emotion Recognition Blockchain System...")
        
        # Initialize blockchain and emotion recognizer
        blockchain = EmotionAudioBlockchain(difficulty=2)
        emotion_recognizer = EmotionRecognizer()  # This will train if no model exists
        
        # Load existing blockchain if available
        blockchain.load_from_file()
        
        while True:
            print("\n=== AI in Emotion Prediction & Healthcare ===")
            print("1. Record new audio")
            print("2. View all recordings")
            print("3. Verify blockchain")
            print("4. View statistics")
            print("5. Exit")
            
            try:
                choice = input("\nEnter your choice (1-5): ").strip()
                
                if choice == '1':
                    record_and_process(blockchain, emotion_recognizer)
                
                elif choice == '2':
                    if len(blockchain.chain) <= 1:
                        print("\nNo recordings found in the blockchain.")
                        continue
                        
                    print("\nStored Recordings:")
                    for block in blockchain.chain[1:]:  # Skip genesis block
                        print(f"\nBlock {block.index}")
                        print(f"Timestamp: {block.timestamp}")
                        print(f"Emotion: {block.emotion}")
                        print(f"Hash: {block.hash}")
                        
                        play_choice = input("Would you like to play this recording? (y/n): ").strip().lower()
                        if play_choice == 'y':
                            print("Playing audio...")
                            play_audio_from_block(block)
                
                elif choice == '3':
                    print("\nVerifying blockchain integrity...")
                    if blockchain.is_chain_valid():
                        print("✅ Blockchain is valid!")
                    else:
                        print("❌ Blockchain validation failed!")
                
                elif choice == '4':
                    stats = blockchain.get_statistics()
                    print("\nBlockchain Statistics:")
                    print(f"Total Recordings: {stats['total_recordings']}")
                    print("\nEmotion Distribution:")
                    for emotion, count in stats['emotions'].items():
                        percentage = (count / stats['total_recordings']) * 100
                        print(f"{emotion}: {count} recordings ({percentage:.1f}%)")
                
                elif choice == '5':
                    print("\nSaving blockchain and exiting...")
                    blockchain.save_to_file()
                    break
                
                else:
                    print("\nInvalid choice. Please try again.")
            
            except KeyboardInterrupt:
                print("\nOperation cancelled by user")
                blockchain.save_to_file()
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                print(f"An error occurred: {str(e)}")
                
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")
        print(f"Failed to start application: {str(e)}")

if __name__ == "__main__":
    main()