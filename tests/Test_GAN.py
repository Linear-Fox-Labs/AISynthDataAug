from models import train

if __name__ == '__main__':
    train(epochs=1, batch_size=42, save_interval=10)